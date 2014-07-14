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

class cuReconData : public basicReconData
{
public:
    cuReconData(int size);

    virtual void addChannelData(const ComplexVector *data) override;
    virtual void addTrajComponent(FloatVector *trajComp) override;
    virtual void setDcf(FloatVector *dcf) override;
    void transformTrajComponent(float translation, float scale, int comp);

    const cuFloatVector *getTrajComponent(int comp) const
    { return m_traj[comp].get(); }

    const cuFloatVector *getDcf() const
    { return m_dcf.get(); }

    const cuComplexVector *getChannelData(int channel) const
    { return m_kDataMultiChannel[channel].get(); }

    std::pair<float, float> getCompBounds(int comp) const
    { return m_bounds[comp]; }

    int channels() const {return m_kDataMultiChannel.size();}
    int rcDim() const { return m_traj.size(); }
    void clear();

private:
    std::vector<std::unique_ptr<const cuComplexVector>> m_kDataMultiChannel;
    std::vector<std::unique_ptr<cuFloatVector>> m_traj;
    std::unique_ptr<cuFloatVector> m_dcf;
};

#endif // CURECONDATA_H
