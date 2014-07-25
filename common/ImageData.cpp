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

template class ImageData<std::vector, float>;
template class ImageData<thrust::device_vector, float>;
