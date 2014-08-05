#ifndef CURECONDATA_H
#define CURECONDATA_H

#include "ReconData.h"

template<typename T>
class cuReconData : public ReconData<thrust::device_vector, T>
{
public:
    typedef typename ReconData<thrust::device_vector, T>::LocalVector cuVector;
    typedef typename ReconData<thrust::device_vector, T>::LocalComplexVector cuComplexVector;

    cuReconData(int size);

    void preprocess(T half_W);

private:
    virtual void transformLocalTraj(float translation, float scale) override;
    void addTrajIndexBlock(cuVector &index);

    std::vector<std::unique_ptr<cuVector>> m_traj_index_blocks;

};

#include "cuReconData.inl"

#endif // CURECONDATA_H
