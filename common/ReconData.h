#ifndef RECONDATA_H
#define RECONDATA_H

#include <memory>
#include <thrust/device_vector.h>
#include "basicReconData.h"

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

void thrust_scale(thrust::device_vector<float> &traj, float translation, float scale);

template<template<typename, typename> class C, typename T>
class ReconData : public basicReconData<T>
{
public:
    using typename basicReconData<T>::Vector;
    using typename basicReconData<T>::ComplexVector;
    typedef typename LocalVectorType<C, T>::type LocalVector;
    typedef typename LocalComplexVectorType<C, T>::type LocalComplexVector;

    ReconData(int size);
    void transformTrajComponent(float translation, float scale, int comp);

    const LocalVector *getTrajComponent(int comp) const {
        return m_traj[comp].get();
    }

    const LocalVector *getDcf() const {
        return m_dcf.get();
    }

    const LocalComplexVector *getChannelData(int channel) const {
        return m_kDataMultiChannel[channel].get();
    }

    int channels() const { return m_kDataMultiChannel.size(); }
    int rcDim() const { return m_traj.size(); }
    void clear();

private:
    using basicReconData<T>::m_bounds;
    using basicReconData<T>::m_size;

    virtual void addData(ComplexVector &data) override;
    virtual void addTraj(Vector &traj) override;
    virtual void addDcf(Vector &dcf) override;

    template<typename V, typename LV>
    LV *toLocalVector(V &v) const;
    void Scale(std::vector<T> &traj, float translation, float scale);
    void Scale(thrust::device_vector<T> &traj, float translation, float scale);

    std::vector<std::unique_ptr<const LocalComplexVector>> m_kDataMultiChannel;
    std::vector<std::unique_ptr<LocalVector>> m_traj;
    std::unique_ptr<LocalVector> m_dcf;
};
#endif // RECONDATA_H
