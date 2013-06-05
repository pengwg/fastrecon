#include <math.h>
#include <gsl/gsl_sf_bessel.h>

#include "ConvKernel.h"

ConvKernel::ConvKernel(float kWidth,  float overGridFactor, int length)
    : m_kWidth(kWidth), m_ogFactor(overGridFactor), m_length(length)
{

}

ConvKernel::~ConvKernel()
{

}

const QVector<float> & ConvKernel::getKernelData()
{
    if (m_kernelData.get() == 0)
        m_kernelData.reset(new QVector<float> (m_length));
    else
        return *m_kernelData;

    float w = m_kWidth;
    float a = m_ogFactor;
    float beta = M_PI * sqrt(w * w / (a * a) * (a - 0.5) * (a - 0.5) - 0.8);

    float dk = w / 2.0 / (m_length -1);
    float kernel0;

    for (int i = 0; i < m_length; i++) {
        float k = dk * i;
        double x = beta * sqrt(1 - powf(2 * k / w, 2));

        (*m_kernelData)[i] = gsl_sf_bessel_I0(x) / w;

        if (i == 0) kernel0 = (*m_kernelData)[0];
        (*m_kernelData)[i] /= kernel0;

    }

    return *m_kernelData;
}


float ConvKernel::getKernelWidth() const
{
    return m_kWidth;
}
