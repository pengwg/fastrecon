#ifndef HOSTIMAGEDATA_H
#define HOSTIMAGEDATA_H

#include "ImageData.h"

template<typename T>
class cuImageData;

template<typename T>
class hostImageData : public ImageData<std::vector, T>
{
public:
    using typename ImageData<std::vector, T>::LocalComplexVector;

    hostImageData();
    hostImageData(const int dim, const ImageSize &imageSize, LocalComplexVector *image = nullptr);

    template<typename T1>
    hostImageData(T1 &&imageData);

    template<typename T1>
    hostImageData<T> &operator=(T1 &&imageData);

    virtual void fftShift() override;
    virtual void lowFilter(int res) override;
    virtual void normalize() override;

private:
    using ImageData<std::vector, T>::m_size;
    using ImageData<std::vector, T>::m_dim;
    using ImageData<std::vector, T>::m_data;
    using ImageData<std::vector, T>::m_channels;

    virtual void copy(const basicImageData &imageData) override;
    virtual void copy(basicImageData &&imageData) override;

    void fftShift2(LocalComplexVector *data);
    void fftShift3(LocalComplexVector *data);

    friend class cuImageData<T>;
};

#include "hostImageData.inl"

#endif // HOSTIMAGEDATA_H
