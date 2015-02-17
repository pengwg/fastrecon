#ifndef COMMON_H
#define COMMON_H

#include <complex>
#include <vector>

typedef struct
{
    unsigned x;
    unsigned y;
    unsigned z;
} ImageSize;

template<typename T>
using hostVector = std::vector<T>;

template<typename T>
using ComplexVector = hostVector<std::complex<T>>;

#ifdef BUILD_CUDA
#include <cuComplex.h>
#include <thrust/device_vector.h>

template<typename T>
using cuVector = typename thrust::device_vector<T>;

template<typename T>
struct cuComplexVectorType
{
};

template<>
struct cuComplexVectorType<float>
{
    typedef thrust::device_vector<cuComplex> type;
};

template<>
struct cuComplexVectorType<double>
{
    typedef thrust::device_vector<cuDoubleComplex> type;
};

template<typename T>
using cuComplexVector = typename cuComplexVectorType<T>::type;
#endif // BUILD_CUDA

#endif // COMMON_H
