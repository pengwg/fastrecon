#ifndef CUIMAGEDATA_H
#define CUIMAGEDATA_H


template<typename T>
class cuImageData : public ImageData<thrust::device_vector, T>
{
public:
    using typename ImageData<thrust::device_vector, T>::LocalComplexVector;

    cuImageData();
    cuImageData(const int dim, const ImageSize &imageSize, LocalComplexVector *image = nullptr);

    template<typename T1>
    cuImageData(T1 &&imageData);

    template<typename T1>
    cuImageData<T> &operator=(T1 &&imageData);

    virtual void fftShift() override;
    virtual void lowFilter(int res) override;
    virtual void normalize() override;

private:
    using ImageData<thrust::device_vector, T>::m_size;
    using ImageData<thrust::device_vector, T>::m_dim;
    using ImageData<thrust::device_vector, T>::m_data;
    using ImageData<thrust::device_vector, T>::m_channels;

    virtual void copy(const basicImageData &imageData) override;
    virtual void copy(basicImageData &&imageData) override;

    void fftShift2(LocalComplexVector *data);
    void fftShift3(LocalComplexVector *data);
};

#include "cuImageData.inl"


#endif // CUIMAGEDATA_H
