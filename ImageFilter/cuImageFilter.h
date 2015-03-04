#ifndef CUIMAGEFILTER_H
#define CUIMAGEFILTER_H

#include "ImageFilter.h"
#include "cuImageData.h"

template<typename T>
class cuImageFilter : public ImageFilter<T>
{
public:
    cuImageFilter(cuImageData<T> &imageData);

    virtual void lowFilter(int res) override;
    virtual void normalize() override;

private:
    cuImageData<T> &m_associatedData;

    virtual void fftShift2(ComplexVector<T> *data) override;
    virtual void fftShift3(ComplexVector<T> *data) override;
};

#endif // CUIMAGEFILTER_H
