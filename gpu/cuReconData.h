#ifndef CURECONDATA_H
#define CURECONDATA_H

#include <memory>
#include <thrust/device_vector.h>
#include "basicReconData.h"

typedef struct __align__(8) {
    float real;
    float imag;
} cuComplexFloat;

typedef thrust::device_vector<cuComplexFloat> cuComplexVector;
typedef thrust::device_vector<float> cuFloatVector;

void thrust_scale(cuFloatVector *traj, float translation, float scale);

template<typename T>
class cuReconData : public basicReconData<T>
{
public:
    using typename basicReconData<T>::Vector;
    using typename basicReconData<T>::ComplexVector;

    cuReconData(int size);

    void transformTrajComponent(float translation, float scale, int comp);

    const cuFloatVector *getTrajComponent(int comp) const
    { return m_traj[comp].get(); }

    const cuFloatVector *getDcf() const
    { return m_dcf.get(); }

    const cuComplexVector *getChannelData(int channel) const
    { return m_kDataMultiChannel[channel].get(); }

    int channels() const {return m_kDataMultiChannel.size();}
    int rcDim() const { return m_traj.size(); }
    void clear();

private:
    using basicReconData<T>::m_bounds;
    using basicReconData<T>::m_size;

    virtual void addData(ComplexVector &data);
    virtual void addTraj(Vector &traj);
    virtual void addDcf(Vector &dcf);

    std::vector<std::unique_ptr<const cuComplexVector>> m_kDataMultiChannel;
    std::vector<std::unique_ptr<cuFloatVector>> m_traj;
    std::unique_ptr<cuFloatVector> m_dcf;
};

#endif // CURECONDATA_H
