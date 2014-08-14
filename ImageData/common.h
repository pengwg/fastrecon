#ifndef COMMON_H
#define COMMON_H

#include <complex>
#include <vector>
#include <cuComplex.h>
#include <thrust/device_vector.h>

template<typename T>
using ComplexVector = std::vector<std::complex<T>>;

template<template<typename, typename> class C, typename T>
struct LocalVectorType {
    typedef std::vector<T> type;
};

template<typename T>
struct LocalVectorType<thrust::device_vector, T> {
    typedef thrust::device_vector<T> type;
};

template<typename T>
struct LocalVectorType<thrust::host_vector, T> {
    typedef thrust::host_vector<T> type;
};

template<template<typename, typename> class C, typename T>
struct LocalComplexVectorType {
    typedef ComplexVector<T> type;
};

template<>
struct LocalComplexVectorType<thrust::device_vector, float> {
    typedef thrust::device_vector<cuComplex> type;
};

template<>
struct LocalComplexVectorType<thrust::device_vector, double> {
    typedef thrust::device_vector<cuDoubleComplex> type;
};

template<>
struct LocalComplexVectorType<thrust::host_vector, float> {
    typedef thrust::host_vector<cuComplex> type;
};

template<>
struct LocalComplexVectorType<thrust::host_vector, double> {
    typedef thrust::host_vector<cuDoubleComplex> type;
};

template<typename T>
using cuVector = typename LocalVectorType<thrust::device_vector, T>::type;

template<typename T>
using cuComplexVector = typename LocalComplexVectorType<thrust::device_vector, T>::type;

#endif // COMMON_H
