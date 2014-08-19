#ifndef COMMON_H
#define COMMON_H

#include <complex>
#include <vector>
#include <cuComplex.h>
#include <thrust/device_vector.h>

template<typename T>
using hostVector = thrust::host_vector<T>;

template<typename T>
using ComplexVector = hostVector<std::complex<T>>;

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

#endif // COMMON_H
