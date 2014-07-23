#include <iostream>
#include <math.h>

#include "ImageData.h"

template<template<typename, typename> class C, typename T>
ImageData<C, T>::ImageData(const int dim, const ImageSize &size, LocalComplexVector *image)
    : basicImageData(dim, size)
{
    if (dim == 2)
        m_size.z = 1;

    addChannelImage(image);
}

// Deep copy
template<template<typename, typename> class C, typename T>
ImageData<C, T>::ImageData(const ImageData &imageData)
{
    copy(imageData);
}

// Move data
template<template<typename, typename> class C, typename T>
ImageData<C, T>::ImageData(ImageData &&imageData)
{
    move(imageData);
}

// Copy
template<template<typename, typename> class C, typename T>
ImageData<C, T> &ImageData<C, T>::operator=(const ImageData &imageData)
{
    copy(imageData);
    return *this;
}

// Move
template<template<typename, typename> class C, typename T>
ImageData<C, T> &ImageData<C, T>::operator=(ImageData &&imageData)
{
    move(imageData);
    return *this;
}

template<template<typename, typename> class C, typename T>
void ImageData<C, T>::addChannelImage(LocalComplexVector *image)
{
    if (image == nullptr) return;

    if (image->size() != dataSize())
    {
        std::cerr << "Error: ImageData wrong size!" << std::endl;
        exit(1);
    }
    m_data.push_back(std::unique_ptr<LocalComplexVector>(image));
    m_channels = m_data.size();
}

template<template<typename, typename> class C, typename T>
const typename ImageData<C, T>::LocalComplexVector *ImageData<C, T>::getChannelImage(int channel) const
{
    if (channel < channels())
        return m_data[channel].get();
    else
        return nullptr;
}

template<template<typename, typename> class C, typename T>
typename ImageData<C, T>::LocalComplexVector *ImageData<C, T>::getChannelImage(int channel)
{
    if (channel < channels())
        return m_data[channel].get();
    else
        return nullptr;
}

template<template<typename, typename> class C, typename T>
void ImageData<C, T>::copy(const basicImageData &imageData)
{
    const ImageData &im = dynamic_cast<const ImageData &>(imageData);
    m_dim = im.m_dim;
    m_size = im.m_size;
    m_data.clear();

    for (const auto &data : im.m_data)
    {
        auto data_copy = new LocalComplexVector(*data);
        addChannelImage(data_copy);
    }

    // std::cout << "Copy called" << std::endl;
}

template<template<typename, typename> class C, typename T>
void ImageData<C, T>::move(basicImageData &imageData)
{
    ImageData &im = dynamic_cast<ImageData &>(imageData);
    m_dim = im.m_dim;
    m_size = im.m_size;
    m_channels = im.m_channels;

    m_data = std::move(im.m_data);

    im.m_dim = 0;
    im.m_size = {0};
    im.m_channels = 0;
    // std::cout << "Move called" << std::endl;
}

template class ImageData<std::vector, float>;
template class ImageData<thrust::device_vector, float>;
