#include <iostream>
#include <math.h>

#include "ImageData.h"

ImageData::ImageData()
    : m_dim(0), m_size({0})
{
}

ImageData::ImageData(const int dim, const ImageSize &size, ComplexVector *image)
    : m_dim(dim), m_size(size)
{
    if (dim == 2)
        m_size.z = 1;

    addChannelImage(image);
}

void ImageData::addChannelImage(ComplexVector *image)
{
    if (image == nullptr) return;

    if (image->size() != dataSize())
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

ImageSize ImageData::imageSize() const
{
    return m_size;
}

int ImageData::dataSize() const
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
#pragma omp parallel for
    for (int n = 0; n < channels(); n++)
    {
        auto data = getChannelImage(n);

        if (m_dim == 3)
            fftShift3(data);
        else
            fftShift2(data);
    }
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

void ImageData::lowFilter(int res)
{
    int x0 = m_size.x / 2;
    int y0 = m_size.y / 2;
    int z0 = m_size.z / 2;
    float att = 2.0 * res * res / 4.0;

    FloatVector coeff;
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

void ImageData::normalize()
{
    FloatVector mag(dataSize(), 0);

    for (const auto &data : m_data)
    {
        auto itMag = mag.begin();
        for (const auto &value : *data)
        {
            *itMag++ += std::norm(value);
        }
    }

    for (auto &data : m_data)
    {
        auto itMag = mag.cbegin();
        for (auto &value : *data)
        {
            value /= sqrtf(*itMag++);
        }
    }
}

ImageData ImageData::makeCopy() const
{
    ImageData image(m_dim, m_size);

    for (const auto &data : m_data)
    {
        auto out = new ComplexVector(*data);
        image.addChannelImage(out);
    }

    return image;
}
