#ifndef CUIMAGEDATA_H
#define CUIMAGEDATA_H

#include "ImageData.h"

template<typename T>
class hostImageData;

template<typename T>
class cuImageData : public ImageData<thrust::host_vector, T>
{
public:
    using typename ImageData<thrust::host_vector, T>::LocalComplexVector;
    typedef typename LocalComplexVectorType<thrust::device_vector, T>::type cuComplexVector;

    cuImageData();
    cuImageData(const int dim, const ImageSize &imageSize, LocalComplexVector *image = nullptr);

    template<typename T1>
    cuImageData(T1 &&imageData);

    template<typename T1>
    cuImageData<T> &operator=(T1 &&imageData);

    using ImageData<thrust::host_vector, T>::addChannelImage;
    void addChannelImage(cuComplexVector *image);
    cuComplexVector *getCUChannelImage(int channel);

    virtual void fftShift() override;
    virtual void lowFilter(int res) override;
    virtual void normalize() override;

private:
    using ImageData<thrust::host_vector, T>::m_size;
    virtual void copy(const basicImageData &imageData) override;
    virtual void copy(basicImageData &&imageData) override;

    void fftShift2(LocalComplexVector *data);
    void fftShift3(LocalComplexVector *data);

    std::unique_ptr<cuComplexVector> m_cudata;
    int m_channel_in_device = 0;

    friend class hostImageData<T>;
};

#include "cuImageData.inl"


#endif // CUIMAGEDATA_H
