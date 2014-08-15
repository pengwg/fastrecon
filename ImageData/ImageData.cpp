#include <iostream>
#include <math.h>

#include "ImageData.h"

template<typename T>
ImageData<T>::ImageData(const int dim, const ImageSize &size, ComplexVector<T> *image)
    : m_dim(dim), m_size(size)
{
    if (dim == 2)
        m_size.z = 1;

    addChannelImage(image);
}

template<typename T>
ImageData<T>::ImageData(const ImageData<T> &imageData)
{
    copy(imageData);
}

template<typename T>
ImageData<T>::ImageData(ImageData<T> &&imageData)
{
    copy(std::move(imageData));
}

// Copy
template<typename T>
ImageData<T> &ImageData<T>::operator=(const ImageData<T> &imageData)
{
    copy(imageData);
    return *this;
}

// Move
template<typename T>
ImageData<T> &ImageData<T>::operator=(ImageData<T> &&imageData)
{
    copy(std::move(imageData));
    return *this;
}

template<typename T>
int ImageData<T>::dataSize() const
{
    if (m_dim == 3)
        return m_size.x * m_size.y * m_size.z;
    else
        return m_size.x * m_size.y;
}

template<typename T>
void ImageData<T>::addChannelImage(ComplexVector<T> *image)
{
    if (image == nullptr) return;

    if (image->size() != dataSize())
    {
        std::cerr << "Error: ImageData wrong size!" << std::endl;
        exit(1);
    }
    m_data_multichannel.push_back(std::unique_ptr<ComplexVector<T>>(image));
    m_channels = m_data_multichannel.size();
}

template<typename T>
const ComplexVector<T> *ImageData<T>::getChannelImage(int channel) const
{
    if (channel < channels())
        return m_data_multichannel[channel].get();
    else
        return nullptr;
}

template<typename T>
ComplexVector<T> *ImageData<T>::getChannelImage(int channel)
{
    if (channel < channels())
        return m_data_multichannel[channel].get();
    else
        return nullptr;
}

template<typename T>
void ImageData<T>::copy(const ImageData<T> &imageData)
{
    m_dim = imageData.dim();
    m_size = imageData.imageSize();
    m_data_multichannel.clear();

    for (const auto &data : imageData.m_data_multichannel)
    {
        auto data_copy = new ComplexVector<T>(*data);
        addChannelImage(data_copy);
    }
    std::cout << "-- ImageData: copy --" << std::endl;
}

template<typename T>
void ImageData<T>::copy(ImageData<T> &&imageData)
{
    m_dim = imageData.m_dim;
    m_size = imageData.m_size;
    m_channels = imageData.m_channels;

    m_data_multichannel = std::move(imageData.m_data_multichannel);

    imageData.m_dim = 0;
    imageData.m_size = {0};
    imageData.m_channels = 0;
    std::cout << "-- ImageData: move --" << std::endl;
}


template<typename T>
void ImageData<T>::fftShift()
{
#pragma omp parallel for
    for (int n = 0; n < channels(); n++)
    {
        auto data = getChannelImage(n);

        if (this->m_dim == 3)
            fftShift3(data);
        else
            fftShift2(data);
    }
}

template<typename T>
void ImageData<T>::fftShift2(ComplexVector<T> *data)
{
    int n0h = m_size.x / 2;
    int n1h = m_size.y / 2;

    int x1, y1;

    for (int y = 0; y < n1h; y++)
    {
        y1 = y + n1h;

        for (int x = 0; x < m_size.x; x++)
        {
            x1 = x < n0h ? x + n0h : x - n0h;
            int i = y * m_size.x + x;
            int j = y1 * m_size.x + x1;

            std::swap(data->at(i), data->at(j));
        }
    }
}

template<typename T>
void ImageData<T>::fftShift3(ComplexVector<T> *data)
{
    int n0h = m_size.x / 2;
    int n1h = m_size.y / 2;
    int n2h = m_size.z / 2;

    int x1, y1, z1;

    for (int z = 0; z < n2h; z++)
    {
        z1 = z + n2h;

        for (int y = 0; y < m_size.y; y++)
        {
            y1 = y < n1h ? y + n1h : y - n1h;

            for (int x = 0; x < m_size.x; x++)
            {
                x1 = x < n0h ? x + n0h : x - n0h;

                int i = z * m_size.x * m_size.y + y * m_size.x + x;
                int j = z1 * m_size.x * m_size.y + y1 * m_size.x + x1;

                std::swap(data->at(i), data->at(j));
            }
        }
    }
}

template<typename T>
void ImageData<T>::lowFilter(int res)
{
    int x0 = m_size.x / 2;
    int y0 = m_size.y / 2;
    int z0 = m_size.z / 2;
    float att = 2.0 * res * res / 4.0;

    std::vector<T> coeff;
    for (int r = 0; r < 2000; r++)
    {
        coeff.push_back(expf(-r / 100.0));
    }

#pragma omp parallel for
    for (int n = 0; n < channels(); n++)
    {
        auto itData = getChannelImage(n)->begin();

        for (int z = 0; z < m_size.z; z++)
        {
            int r1 = (z - z0) * (z - z0);
            for (int y = 0; y < m_size.y; y++)
            {
                int r2 = (y - y0) * (y - y0) + r1;
                for (int x = 0; x < m_size.x; x++)
                {
                    int r = (x - x0) * (x - x0) + r2;
                    int idx = (int)(r / att * 100.0);
                    if (idx >= 2000)
                        *itData++ = 0;
                    else
                        *itData++ *= coeff[idx];
                }
            }
        }
    }
}

template<typename T>
void ImageData<T>::normalize()
{
    std::vector<T> mag(dataSize(), 0);

    for (const auto &data : m_data_multichannel)
    {
        auto itMag = mag.begin();
        for (const auto &value : *data)
        {
            *itMag++ += std::norm(value);
        }
    }

    for (auto &data : m_data_multichannel)
    {
        auto itMag = mag.cbegin();
        for (auto &value : *data)
        {
            value /= sqrtf(*itMag++);
        }
    }
}

template class ImageData<float>;
