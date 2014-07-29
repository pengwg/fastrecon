#include <cassert>
#include "hostImageData.h"

template<typename T>
cuImageData<T>::cuImageData()
{
}

template<typename T>
cuImageData<T>::cuImageData(const int dim, const ImageSize &imageSize, LocalComplexVector *image)
    : ImageData<thrust::host_vector, T>(dim, imageSize, image)
{
}

template<typename T>
template<typename T1>
cuImageData<T>::cuImageData(T1 &&imageData)
{
    copy(std::forward<T1>(imageData));
}

template<typename T>
template<typename T1>
cuImageData<T> &cuImageData<T>::operator=(T1 &&imageData)
{
    copy(std::forward<T1>(imageData));
    return *this;
}

template<typename T>
void cuImageData<T>::copy(const basicImageData &imageData)
{
    ImageData<thrust::host_vector, T>::copy(imageData);
    auto im = dynamic_cast<const hostImageData<T> *>(&imageData);
    if (im)
    {
        for (const auto &data : im->m_data)
        {
            auto h_data = reinterpret_cast<std::vector<typename LocalComplexVector::value_type> &>(*data);
            auto data_copy = new LocalComplexVector(h_data);
            this->addChannelImage(data_copy);
        }
        std::cout << "-- Copy host to cuda --" << std::endl;
    }
}

template<typename T>
void cuImageData<T>::copy(basicImageData &&imageData)
{
    ImageData<thrust::host_vector, T>::copy(std::move(imageData));
}

template<typename T>
void cuImageData<T>::fftShift()
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
void cuImageData<T>::fftShift2(LocalComplexVector  *data)
{

}

template<typename T>
void cuImageData<T>::fftShift3(LocalComplexVector *data)
{

}

template<typename T>
void cuImageData<T>::lowFilter(int res)
{

}

template<typename T>
void cuImageData<T>::normalize()
{

}

//template class cuImageData<float>;
