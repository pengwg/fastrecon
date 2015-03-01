#ifndef CUIMAGEFILTER_H
#define CUIMAGEFILTER_H

#include "ImageFilter.h"
#include "cuImageData.h"

template<typename T>
class cuImageFilter : public ImageFilter<T>
{
public:
    cuImageFilter(cuImageData<T> &imageData);

private:
    cuImageData<T> m_associatedData;
};

#endif // CUIMAGEFILTER_H
