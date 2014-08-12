#ifndef COMMON_H
#define COMMON_H

#include <complex>
#include <vector>
#include <cuComplex.h>
#include <thrust/device_vector.h>

typedef std::vector<float> FloatVector;
typedef std::vector<std::complex<float>> ComplexVector;

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
    typedef std::vector<std::complex<T>> type;
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

#endif // COMMON_H
