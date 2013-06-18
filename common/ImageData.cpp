#include <iostream>
#include "ImageData.h"

ImageData::ImageData(const int dim, const ImageSize &size)
    : m_dim(dim), m_size(size)
{
    if (dim == 2)
        m_size.z = 1;
}

void ImageData::addChannelImage(ComplexVector *image)
{
    if (image->size() != length())
    {
        std::cerr << "Error: ImageData wrong size!" << std::endl;
        exit(1);
    }
    m_data.push_back(std::shared_ptr<ComplexVector>(image));
}

const ComplexVector *ImageData::getChannelImage(int channel) const
{
    if (channel < channels())
        return m_data[channel].get();
    else
        return nullptr;
}

ComplexVector *ImageData::getChannelImage(int channel)
{
    if (channel < channels())
        return m_data[channel].get();
    else
        return nullptr;
}

ImageSize ImageData::size() const
{
    return m_size;
}

int ImageData::length() const
{
    if (m_dim == 3)
        return m_size.x * m_size.y * m_size.z;
    else
        return m_size.x * m_size.y;
}

int ImageData::channels() const
{
    return m_data.size();
}


int ImageData::dim() const
{
    return m_dim;
}

void ImageData::fftShift()
{
    for (int n = 0; n < channels(); n++)
    {
        auto data = getChannelImage(n);

        if (m_dim == 3)
            fftShift3(data);
        else
            fftShift2(data);
    }
}

ImageData ImageData::crop_sos(ImageSize size) const
{
    ImageData img(m_dim, m_size);
    auto out = new ComplexVector(length());

    for (int n = 0; n < channels(); n++)
    {
        auto itOut = out->begin();
        auto itInput = getChannelImage(n)->cbegin();
#pragma omp parallel for
        for (int i = 0; i < length(); i++)
        {
            auto data = *(itInput + i);
            *(itOut+i) += data * std::conj(data);
        }
    }

    for (auto &data : *out)
    {
        data = std::sqrt(data);
    }

    img.addChannelImage(out);
    return img;
}

void ImageData::fftShift2(ComplexVector *data)
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

void ImageData::fftShift3(ComplexVector *data)
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
