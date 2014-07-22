#ifndef COMMON_H
#define COMMON_H

#include <complex>
#include <vector>
#include <cuComplex.h>
#include <thrust/device_vector.h>

template<template<typename, typename> class C, typename T, typename... A>
struct LocalVectorType {
    typedef C<T, A...> type;
};

template<template<typename, typename> class C, typename T, typename... A>
struct LocalComplexVectorType {
    typedef C<std::complex<T>, A...> type;
};

template<typename... A>
struct LocalComplexVectorType<thrust::device_vector, float, A...> {
    typedef thrust::device_vector<cuComplex> type;
};

template<typename... A>
struct LocalComplexVectorType<thrust::device_vector, double, A...> {
    typedef thrust::device_vector<cuDoubleComplex> type;
};

template<typename T>
using hostComplexVector = typename LocalComplexVectorType<std::vector, T>::type;

template<typename T>
using cuComplexVector = typename LocalComplexVectorType<thrust::device_vector, T>::type;

typedef std::vector<std::complex<float>> ComplexVector;

#endif // COMMON_H
