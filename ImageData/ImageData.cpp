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
void ImageData<T>::updateChannelImage(ComplexVector<T> &&image, const ImageSize &imageSize, int channel)
{
    if (image.size() != imageSize.x * imageSize.y * imageSize.z)
    {
        std::cerr << "Error: ImageData update wrong size!" << std::endl;
        exit(1);
    }

    *getChannelImage(channel) = std::move(image);
    m_size = imageSize;
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

template class ImageData<float>;
