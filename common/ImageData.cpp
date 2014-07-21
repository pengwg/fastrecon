#include <iostream>
#include <math.h>

#include "ImageData.h"

template<template<typename, typename> class C, typename T>
ImageData<C, T>::ImageData()
{
}

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
void ImageData<C, T>::fftShift()
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

template<template<typename, typename> class C, typename T>
void ImageData<C, T>::fftShift2(std::vector<std::complex<T>> *data)
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

template<template<typename, typename> class C, typename T>
void ImageData<C, T>::fftShift3(std::vector<std::complex<T>> *data)
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

template<template<typename, typename> class C, typename T>
void ImageData<C, T>::lowFilter(int res)
{
    int x0 = m_size.x / 2;
    int y0 = m_size.y / 2;
    int z0 = m_size.z / 2;
    float att = 2.0 * res * res / 4.0;

    LocalVector coeff;
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

template<template<typename, typename> class C, typename T>
void ImageData<C, T>::normalize()
{
    LocalVector mag(dataSize(), 0);

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
//template class ImageData<thrust::device_vector, float>;
