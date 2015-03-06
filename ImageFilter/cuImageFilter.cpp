#include "cuImageFilter.h"

template<typename T>
cuImageFilter<T>::cuImageFilter(cuImageData<T> &imageData)
    : ImageFilter<T>(imageData), m_associatedData(imageData)
{
}

template<typename T>
void cuImageFilter<T>::lowFilter(int res)
{
    ImageFilter<T>::lowFilter(res);
    m_associatedData.invalidateDevice();
}

template<typename T>
void cuImageFilter<T>::normalize()
{
    ImageFilter<T>::normalize();
    m_associatedData.invalidateDevice();
}

template<typename T>
void cuImageFilter<T>::fftPlan(int sign)
{
    if (this->m_fft != nullptr)
        delete this->m_fft;

    switch (sign)
    {
    case FFTW_FORWARD: sign = CUFFT_FORWARD;
        break;
    case FFTW_BACKWARD: sign = CUFFT_INVERSE;
    }

    this->m_fft = new cuFFT(m_associatedData.dim(), m_associatedData.imageSize(), sign);
    this->m_fft->plan();
}

template<typename T>
void cuImageFilter<T>::fftShift2(ComplexVector<T> *data)
{
    ImageFilter<T>::fftShift2(data);
    m_associatedData.invalidateDevice();
}

template<typename T>
void cuImageFilter<T>::fftShift3(ComplexVector<T> *data)
{
    ImageFilter<T>::fftShift3(data);
    m_associatedData.invalidateDevice();
}

template class cuImageFilter<float>;
