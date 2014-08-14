#ifndef RECONDATA_H
#define RECONDATA_H

#include <memory>
#include "common.h"
#include "basicReconData.h"

template<template<typename, typename> class C, typename T>
class ReconData : public basicReconData<T>
{
public:
    using typename basicReconData<T>::TrajVector;

    typedef typename LocalVectorType<C, T>::type LocalVector;
    typedef typename LocalVectorType<C, Point<T>>::type LocalTrajVector;
    typedef typename LocalComplexVectorType<C, T>::type LocalComplexVector;

    const LocalTrajVector *getTraj() const {
        return m_traj.get();
    }

    const LocalVector *getDcf() const {
        return m_dcf.get();
    }

    const LocalComplexVector *getChannelData(int channel) const {
        return m_kDataMultiChannel[channel].get();
    }

    int channels() const { return m_kDataMultiChannel.size(); }
    void clear();

protected:
    ReconData(int size);
    virtual ~ReconData() {}

    using basicReconData<T>::m_bounds;
    using basicReconData<T>::m_size;

    virtual void addData(ComplexVector<T> &data) override final;
    virtual void addTraj(TrajVector &traj) override final;
    virtual void addDcf(std::vector<T> &dcf) override final;

    template<typename V, typename LV>
    LV *toLocalVector(V &v) const;

    std::unique_ptr<LocalTrajVector> m_traj;
    std::vector<std::unique_ptr<const LocalComplexVector>> m_kDataMultiChannel;
    std::unique_ptr<LocalVector> m_dcf;
};
#endif // RECONDATA_H
