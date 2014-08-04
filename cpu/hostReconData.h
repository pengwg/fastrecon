#ifndef HOSTRECONDATA_H
#define HOSTRECONDATA_H

#include "ReconData.h"

template<typename T>
class hostReconData : public ReconData<std::vector, T>
{
public:
    hostReconData(int size);

private:
    virtual void transformLocalTraj(float translation, float scale) override;
};

#include "hostReconData.inl"

#endif // HOSTRECONDATA_H
