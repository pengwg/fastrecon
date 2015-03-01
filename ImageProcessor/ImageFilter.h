#ifndef IMAGEFILTER_H
#define IMAGEFILTER_H

#include "ImageData.h"

template<typename T>
class ImageFilter
{
public:
    static std::shared_ptr<ImageFilter<T>> Create(ImageData<T> &imageData);

    void fftShift();
    virtual void lowFilter(int res);
    virtual void normalize();
    virtual void crop(const ImageSize &imageSize);

protected:
    ImageFilter(ImageData<T> &imageData);

private:
    ImageData<T> &m_associatedData;

    virtual void fftShift2(ComplexVector<T> *data);
    virtual void fftShift3(ComplexVector<T> *data);
};

#endif // IMAGEFILTER_H
