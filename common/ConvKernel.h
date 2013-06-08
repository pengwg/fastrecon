#ifndef CONVKERNEL_H
#define CONVKERNEL_H

#include <memory>
#include <vector>

class ConvKernel
{
public:
    ConvKernel(float kWidth, float overGridFactor, int length = 32);
    ~ConvKernel();

    const std::vector<float> *getKernelData();
    float getKernelWidth() const;

private:
    float m_kWidth;
    float m_ogFactor;
    int m_length;

    std::shared_ptr<std::vector<float> > m_kernelData;
};

#endif // CONVKERNEL_H
