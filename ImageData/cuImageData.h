#ifndef CUIMAGEDATA_H
#define CUIMAGEDATA_H

#include "common.h"
#include "ImageData.h"

template<typename T>
class cuImageData : public ImageData<T>
{
public:
    cuImageData();
    cuImageData(const int dim, const ImageSize &imageSize, cuComplexVector<T> *image = nullptr);

    cuImageData(const cuImageData<T> &imageData);
    cuImageData(cuImageData<T> &&imageData);
    cuImageData<T> &operator=(const cuImageData<T> &imageData);
    cuImageData<T> &operator=(cuImageData<T> &&imageData);

    using ImageData<T>::addChannelImage;
    void addChannelImage(cuComplexVector<T> *image);
    cuComplexVector<T> *cuGetChannelImage(int channel);
    void update();

    virtual void fftShift() override;
    virtual void lowFilter(int res) override;
    virtual void normalize() override;

private:
    using ImageData<T>::m_size;
    virtual void copy(const ImageData<T> &imageData) override;
    virtual void copy(ImageData<T> &&imageData) override;

    void fftShift2(cuComplexVector<T> *data);
    void fftShift3(cuComplexVector<T> *data);

    std::unique_ptr<cuComplexVector<T>> m_cu_data;
    int m_channel_in_device = -1;
};

#endif // CUIMAGEDATA_H
