#include "cuImageData.h"

template<typename T>
cuImageData<T>::cuImageData()
{
}

template<typename T>
cuImageData<T>::cuImageData(const int dim, const ImageSize &imageSize, LocalComplexVector *image)
    : ImageData<thrust::device_vector, T>(dim, imageSize, image)
{
}

template<typename T>
template<typename T1>
cuImageData<T>::cuImageData(T1 &&imageData)
    : ImageData<thrust::device_vector, T>(std::forward<T1>(imageData))
{
}

template<typename T>
template<typename T1>
cuImageData<T> &cuImageData<T>::operator=(T1 &&imageData)
{
    ImageData<thrust::device_vector, T>::operator=(std::forward<T1>(imageData));
    return *this;
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
