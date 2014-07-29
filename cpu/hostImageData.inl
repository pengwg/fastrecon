#include <cassert>
#include "cuImageData.h"

template<typename T>
hostImageData<T>::hostImageData()
{
}

template<typename T>
hostImageData<T>::hostImageData(const int dim, const ImageSize &imageSize, LocalComplexVector *image)
    : ImageData<std::vector, T>(dim, imageSize, image)
{
}

template<typename T>
template<typename T1>
hostImageData<T>::hostImageData(T1 &&imageData)
{
    copy(std::forward<T1>(imageData));
}

template<typename T>
template<typename T1>
hostImageData<T> &hostImageData<T>::operator=(T1 &&imageData)
{
    copy(std::forward<T1>(imageData));
    return *this;
}

template<typename T>
void hostImageData<T>::copy(const basicImageData &imageData)
{
    ImageData<std::vector, T>::copy(imageData);
    auto im = dynamic_cast<const cuImageData<T> *>(&imageData);
    if (im)
    {
        for (const auto &data : im->m_data)
        {
            auto p_data = new LocalComplexVector(m_size.x * m_size.y * m_size.z);
            std::memcpy(p_data->data(), data->data(), sizeof(typename LocalComplexVector::value_type) * p_data->size());
            this->addChannelImage(p_data);
        }
        std::cout << "-- Copy cuda to host --" << std::endl;
    }
}

template<typename T>
void hostImageData<T>::copy(basicImageData &&imageData)
{
    ImageData<std::vector, T>::copy(std::move(imageData));
}

template<typename T>
void hostImageData<T>::fftShift()
{
#pragma omp parallel for
    for (int n = 0; n < this->channels(); n++)
    {
        auto data = this->getChannelImage(n);

        if (this->m_dim == 3)
            fftShift3(data);
        else
            fftShift2(data);
    }
}

template<typename T>
void hostImageData<T>::fftShift2(LocalComplexVector  *data)
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
void hostImageData<T>::fftShift3(LocalComplexVector *data)
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
void hostImageData<T>::lowFilter(int res)
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
    for (int n = 0; n < this->channels(); n++)
    {
        auto itData = this->getChannelImage(n)->begin();

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
void hostImageData<T>::normalize()
{
    std::vector<T> mag(this->dataSize(), 0);

    for (const auto &data : this->m_data)
    {
        auto itMag = mag.begin();
        for (const auto &value : *data)
        {
            *itMag++ += std::norm(value);
        }
    }

    for (auto &data : this->m_data)
    {
        auto itMag = mag.cbegin();
        for (auto &value : *data)
        {
            value /= sqrtf(*itMag++);
        }
    }
}

//template class hostImageData<float>;
