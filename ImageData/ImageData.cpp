#include <iostream>
#include <math.h>

#include "ImageData.h"

template<typename T>
ImageData<T>::ImageData(const int dim, const ImageSize &size, std::unique_ptr<ComplexVector<T>> image)
    : m_dim(dim), m_size(size)
{
    if (dim == 2)
        m_size.z = 1;

    addChannelImage(std::move(image));
}

template<typename T>
ImageData<T>::ImageData(const ImageData<T> &imageData)
{
    copy(imageData);
}

template<typename T>
ImageData<T>::ImageData(ImageData<T> &&imageData)
{
    move(imageData);
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
    move(imageData);
    return *this;
}

template<typename T>
std::size_t ImageData<T>::dataSize() const
{
    if (m_dim == 3)
        return m_size.x * m_size.y * m_size.z;
    else
        return m_size.x * m_size.y;
}

template<typename T>
void ImageData<T>::addChannelImage(std::unique_ptr<ComplexVector<T>> image)
{
    if (image == nullptr) return;

    if (image->size() != dataSize())
    {
        std::cerr << "Error: ImageData wrong size!" << std::endl;
        exit(1);
    }
    m_data_multichannel.push_back(std::move(image));
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
        addChannelImage(std::unique_ptr<ComplexVector<T>>(data_copy));
    }
    std::cout << "-- ImageData: copy --" << std::endl;
}

template<typename T>
void ImageData<T>::move(ImageData<T> &imageData)
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
    auto n0h = m_size.x / 2;
    auto n1h = m_size.y / 2;

    for (auto y = 0ul; y < n1h; y++)
    {
        auto y1 = y + n1h;

        for (auto x = 0ul; x < m_size.x; x++)
        {
            auto x1 = x < n0h ? x + n0h : x - n0h;
            auto i = y * m_size.x + x;
            auto j = y1 * m_size.x + x1;

            std::swap((*data)[i], (*data)[j]);
        }
    }
}

template<typename T>
void ImageData<T>::fftShift3(ComplexVector<T> *data)
{
    auto n0h = m_size.x / 2;
    auto n1h = m_size.y / 2;
    auto n2h = m_size.z / 2;

    for (auto z = 0ul; z < n2h; z++)
    {
        auto z1 = z + n2h;

        for (auto y = 0ul; y < m_size.y; y++)
        {
            auto y1 = y < n1h ? y + n1h : y - n1h;

            for (auto x = 0ul; x < m_size.x; x++)
            {
                auto x1 = x < n0h ? x + n0h : x - n0h;

                auto i = z * m_size.x * m_size.y + y * m_size.x + x;
                auto j = z1 * m_size.x * m_size.y + y1 * m_size.x + x1;

                std::swap((*data)[i], (*data)[j]);
            }
        }
    }
}

template<typename T>
void ImageData<T>::lowFilter(int res)
{
    auto x0 = m_size.x / 2;
    auto y0 = m_size.y / 2;
    auto z0 = m_size.z / 2;
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

        for (auto z = 0ul; z < m_size.z; z++)
        {
            int r1 = (z - z0) * (z - z0);
            for (auto y = 0ul; y < m_size.y; y++)
            {
                int r2 = (y - y0) * (y - y0) + r1;
                for (auto x = 0ul; x < m_size.x; x++)
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

template<typename T>
void ImageData<T>::crop(const ImageSize &imageSize)
{
    auto x0 = (m_size.x - imageSize.x) / 2;
    auto y0 = (m_size.y - imageSize.y) / 2;
    auto z0 = (m_size.z - imageSize.z) / 2;

    if (x0 < 0 || y0 < 0 || z0 < 0) {
        std::cerr << "Crop size larger than image" << std::endl;
        return;
    }

    for (int n = 0; n < channels(); n++)
    {
        auto out = new ComplexVector<float>(imageSize.x * imageSize.y * imageSize.z);
        auto itOut = out->begin();
        auto itInput = getChannelImage(n)->cbegin();

#pragma omp parallel for
        for (auto z = 0ul; z < imageSize.z; z++)
        {
            auto in1 = (z + z0) * (m_size.x * m_size.y) + y0 * m_size.x;
            auto out1 = z * (imageSize.x * imageSize.y);

            for (auto y = 0ul; y < imageSize.y; y++)
            {
                auto in2 = y * m_size.x + in1 + x0;
                auto out2 = y * imageSize.x + out1;
                for (auto x = 0ul; x < imageSize.x; x++)
                {
                    auto in3 = x + in2;
                    auto out3 = x + out2;
                    *(itOut+out3) = *(itInput + in3);
                }
            }
        }
        m_data_multichannel[n].reset(out);
    }
    m_size = imageSize;
}

template class ImageData<float>;
