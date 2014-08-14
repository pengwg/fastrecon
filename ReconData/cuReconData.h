#ifndef CURECONDATA_H
#define CURECONDATA_H

#include "ReconData.h"

template<typename T>
class cuReconData : public ReconData<thrust::device_vector, T>
{
public:
    cuReconData(int size);

private:
    virtual void transformLocalTraj(T translation, T scale) override;
    void addTrajIndexBlock(cuVector<T> &index);

    std::vector<std::unique_ptr<cuVector<T>>> m_traj_index_blocks;
};

#include "cuReconData.inl"

#endif // CURECONDATA_H
