#ifndef CURECONDATA_H
#define CURECONDATA_H

#include "ReconData.h"

template<typename T>
class cuReconData : public ReconData<thrust::device_vector, T>
{
public:
    cuReconData(int size);

private:
    virtual void transformLocalTrajComp(float translation, float scale, int comp) override;
};

#include "cuReconData.inl"

#endif // CURECONDATA_H
