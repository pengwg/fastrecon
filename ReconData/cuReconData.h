#ifndef CURECONDATA_H
#define CURECONDATA_H

#include "ReconData.h"

template<typename T>
class cuReconData : public ReconData<T>
{
public:
    cuReconData(int size);

private:
    virtual void transformLocalTraj(T translation, T scale) override;
    void addTrajIndexBlock(cuVector<T> &index);

    std::unique_ptr<TrajVector> m_traj;
    std::vector<std::unique_ptr<const ComplexVector<T>>> m_kDataMultiChannel;
    std::unique_ptr<std::vector<T>> m_dcf;
};

#include "cuReconData.inl"

#endif // CURECONDATA_H
