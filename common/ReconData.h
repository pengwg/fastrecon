#ifndef RECONDATA_H
#define RECONDATA_H

#include <memory>
#include "common.h"
#include "basicReconData.h"

template<template<typename, typename> class C, typename T>
class ReconData : public basicReconData<T>
{
public:
    using typename basicReconData<T>::Vector;
    using typename basicReconData<T>::ComplexVector;
    typedef typename LocalVectorType<C, T>::type LocalVector;
    typedef typename LocalComplexVectorType<C, T>::type LocalComplexVector;

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

protected:
    ReconData(int size);
    virtual ~ReconData() {}

    using basicReconData<T>::m_bounds;
    using basicReconData<T>::m_size;

    virtual void addData(ComplexVector &data) override final;
    virtual void addTraj(Vector &traj) override final;
    virtual void addDcf(Vector &dcf) override final;

    template<typename V, typename LV>
    LV *toLocalVector(V &v) const;

    std::vector<std::unique_ptr<LocalVector>> m_traj;
    std::vector<std::unique_ptr<const LocalComplexVector>> m_kDataMultiChannel;
    std::unique_ptr<LocalVector> m_dcf;
};
#endif // RECONDATA_H
