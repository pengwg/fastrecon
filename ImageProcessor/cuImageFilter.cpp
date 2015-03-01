#include "cuImageFilter.h"

template<typename T>
cuImageFilter<T>::cuImageFilter(cuImageData<T> &imageData)
    : ImageFilter<T>(imageData), m_associatedData(imageData)
{
}
