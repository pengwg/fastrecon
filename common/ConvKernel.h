#ifndef CONVKERNEL_H
#define CONVKERNEL_H

#include <QVector>
#include <memory>

class ConvKernel
{
public:
    ConvKernel(float kWidth, float overGridFactor, int length = 32);
    ~ConvKernel();

    const QVector<float> & getKernelData();
    float getKernelWidth() const;

private:
    float m_kWidth;
    float m_ogFactor;
    int m_length;

    std::shared_ptr<QVector<float> > m_kernelData;
};

#endif // CONVKERNEL_H
