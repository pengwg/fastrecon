#include <iostream>

#include "ImageFilter.h"
#ifdef BUILD_CUDA
#include "cuImageFilter.h"
#endif // BUILD_CUDA

template<typename T>
std::shared_ptr<ImageFilter<T>> ImageFilter<T>::Create(ImageData<T> &imageData)
{
    ImageFilter<T> *instance = nullptr;

#ifdef BUILD_CUDA
    auto cu_data = dynamic_cast<cuImageData<T> *>(&imageData);
    if (cu_data != nullptr)
        instance = new cuImageFilter<T>(*cu_data);
#endif // BUILD_CUDA
    if (instance == nullptr)
        instance = new ImageFilter<T>(imageData);

    return std::shared_ptr<ImageFilter<T>>(instance);;
}

template<typename T>
ImageFilter<T>::ImageFilter(ImageData<T> &imageData)
    : m_associatedData(imageData)
{
}


template<typename T>
void ImageFilter<T>::fftShift()
{
#pragma omp parallel for
    for (int n = 0; n < m_associatedData.channels(); n++)
    {
        auto data = m_associatedData.getChannelImage(n);

        if (m_associatedData.dim() == 3)
            fftShift3(data);
        else
            fftShift2(data);
    }
}

template<typename T>
void ImageFilter<T>::fftShift2(ComplexVector<T> *data)
{
    auto size = m_associatedData.imageSize();
    auto n0h = size.x / 2;
    auto n1h = size.y / 2;

    for (auto y = 0ul; y < n1h; y++)
    {
        auto y1 = y + n1h;

        for (auto x = 0ul; x < size.x; x++)
        {
            auto x1 = x < n0h ? x + n0h : x - n0h;
            auto i = y * size.x + x;
            auto j = y1 * size.x + x1;

            std::swap((*data)[i], (*data)[j]);
        }
    }
}

template<typename T>
void ImageFilter<T>::fftShift3(ComplexVector<T> *data)
{
    auto size = m_associatedData.imageSize();
    auto n0h = size.x / 2;
    auto n1h = size.y / 2;
    auto n2h = size.z / 2;

    for (auto z = 0ul; z < n2h; z++)
    {
        auto z1 = z + n2h;

        for (auto y = 0ul; y < size.y; y++)
        {
            auto y1 = y < n1h ? y + n1h : y - n1h;

            for (auto x = 0ul; x < size.x; x++)
            {
                auto x1 = x < n0h ? x + n0h : x - n0h;

                auto i = z * size.x * size.y + y * size.x + x;
                auto j = z1 * size.x * size.y + y1 * size.x + x1;

                std::swap((*data)[i], (*data)[j]);
            }
        }
    }
}

template<typename T>
void ImageFilter<T>::lowFilter(int res)
{
    auto size = m_associatedData.imageSize();
    auto x0 = size.x / 2;
    auto y0 = size.y / 2;
    auto z0 = size.z / 2;
    float att = 2.0 * res * res / 4.0;

    std::vector<T> coeff;
    for (int r = 0; r < 2000; r++)
    {
        coeff.push_back(expf(-r / 100.0));
    }

#pragma omp parallel for
    for (int n = 0; n < m_associatedData.channels(); n++)
    {
        auto itData = m_associatedData.getChannelImage(n)->begin();

        for (auto z = 0ul; z < size.z; z++)
        {
            int r1 = (z - z0) * (z - z0);
            for (auto y = 0ul; y < size.y; y++)
            {
                int r2 = (y - y0) * (y - y0) + r1;
                for (auto x = 0ul; x < size.x; x++)
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
void ImageFilter<T>::normalize()
{
    std::vector<T> mag(m_associatedData.dataSize(), 0);

    for (int n = 0; n < m_associatedData.channels(); n++)
    {
        auto data = m_associatedData.getChannelImage(n);
        auto itMag = mag.begin();
        for (const auto &value : *data) {
            *itMag++ += std::norm(value);
        }
    }

    for (int n = 0; n < m_associatedData.channels(); n++)
    {
        auto data = m_associatedData.getChannelImage(n);
        auto itMag = mag.cbegin();
        for (auto &value : *data)
        {
            value /= sqrtf(*itMag++);
        }
    }
}

template<typename T>
void ImageFilter<T>::crop(const ImageSize &imageSize)
{
    auto size = m_associatedData.imageSize();
    auto x0 = (size.x - imageSize.x) / 2;
    auto y0 = (size.y - imageSize.y) / 2;
    auto z0 = (size.z - imageSize.z) / 2;

    if (x0 < 0 || y0 < 0 || z0 < 0) {
        std::cerr << "Crop size larger than image" << std::endl;
        return;
    }

    ComplexVector<T> out;
    ImageData<T> img(m_associatedData.dim(), imageSize);
    for (int n = 0; n < m_associatedData.channels(); n++)
    {
        out.resize(imageSize.x * imageSize.y * imageSize.z);
        auto itOut = out.begin();
        auto itInput = m_associatedData.getChannelImage(n)->cbegin();

#pragma omp parallel for
        for (auto z = 0ul; z < imageSize.z; z++)
        {
            auto in1 = (z + z0) * (size.x * size.y) + y0 * size.x;
            for (auto y = 0ul; y < imageSize.y; y++)
            {
                auto in2 = y * size.x + in1 + x0;
                for (auto x = 0ul; x < imageSize.x; x++)
                {
                    auto in3 = x + in2;
                    *(itOut++) = *(itInput + in3);
                }
            }
        }
        img.addChannelImage(std::move(out));
    }
    m_associatedData = std::move(img);
}

template<typename T>
void ImageFilter<T>::SOS(ImageSize reconSize)
{
    SOS(m_associatedData, reconSize);

    for (auto &data : *m_associatedData.getChannelImage(0))
    {
        data = std::sqrt(data);
    }
}

template<typename T>
void ImageFilter<T>::SOS(const ImageData<T> &map, ImageSize reconSize)
{
    auto imageSize = m_associatedData.imageSize();
    auto x0 = (imageSize.x - reconSize.x) / 2;
    auto y0 = (imageSize.y - reconSize.y) / 2;
    auto z0 = (imageSize.z - reconSize.z) / 2;

    ComplexVector<T> out(reconSize.x * reconSize.y * reconSize.z);
    ImageData<T> img(m_associatedData.dim(), reconSize);

    for (int n = 0; n < m_associatedData.channels(); n++)
    {
        auto itOut = out.begin();
        auto itInput = m_associatedData.getChannelImage(n)->cbegin();
        auto itMap = map.getChannelImage(n)->cbegin();

#pragma omp parallel for
        for (auto z = 0u; z < reconSize.z; z++)
        {
            auto in1 = (z + z0) * (imageSize.x * imageSize.y) + y0 * imageSize.x;
            auto out1 = z * (reconSize.x * reconSize.y);
            for (auto y = 0u; y < reconSize.y; y++)
            {
                auto in2 = y * imageSize.x + in1 + x0;
                auto out2 = y * reconSize.x + out1;
                for (auto x = 0u; x < reconSize.x; x++)
                {
                    auto in3 = x + in2;
                    auto out3 = x + out2;

                    auto data = *(itInput + in3);
                    auto mapData = *(itMap + in3);
                    *(itOut+out3) += data * std::conj(mapData);
                }
            }
        }
    }
    img.addChannelImage(std::move(out));
    m_associatedData = std::move(img);
}

template class ImageFilter<float>;
