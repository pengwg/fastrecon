#ifndef CONVKERNEL_H
#define CONVKERNEL_H

#include <memory>
#include "basicReconData.h"

typedef std::vector<float> FloatVector;
typedef std::vector<std::complex<float>> ComplexVector;

class ConvKernel
{
public:
    ConvKernel(float kWidth, float overGridFactor, int length = 32);
    ~ConvKernel();

    const FloatVector *getKernelData() const;
    float getKernelWidth() const;

private:
    float m_kWidth;
    float m_ogFactor;
    int m_length;

    std::shared_ptr<FloatVector> m_kernelData;
};

#endif // CONVKERNEL_H
