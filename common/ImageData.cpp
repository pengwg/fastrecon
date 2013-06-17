#include <iostream>
#include "ImageData.h"

ImageData::ImageData(const int dim, const ImageSize &size)
    : m_dim(dim), m_size(size)
{

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

ComplexVector *ImageData::getChannelImage(int channel) const
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

