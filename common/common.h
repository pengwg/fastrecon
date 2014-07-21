#ifndef COMMON_H
#define COMMON_H

#include <complex>
#include <vector>
#include <thrust/device_vector.h>

template<template<typename, typename> class C, typename T, typename... A>
struct LocalVectorType {
    typedef C<T, A...> type;
};

template<typename T>
struct cuComplex
{
    T real;
    T imag;
};

template<template<typename, typename> class C, typename T, typename... A>
struct LocalComplexVectorType {
    typedef C<std::complex<T>, A...> type;
};

template<typename T, typename... A>
struct LocalComplexVectorType<thrust::device_vector, T, A...> {
    typedef thrust::device_vector<cuComplex<T>> type;
};

typedef std::vector<std::complex<float>> ComplexVector;

#endif // COMMON_H
