#include <math.h>
#include <memory>
#include <gsl/gsl_sf_bessel.h>

#include "ConvKernel.h"

ConvKernel::ConvKernel(float kWidth,  float overGridFactor, int length)
    : m_kWidth(kWidth), m_ogFactor(overGridFactor), m_length(length)
{
    std::vector<float> *data = new std::vector<float>(m_length);

    float w = m_kWidth;
    float a = m_ogFactor;
    float beta = M_PI * sqrt(w * w / (a * a) * (a - 0.5) * (a - 0.5) - 0.8);

    float dk = w / 2.0 / (m_length -1);
    float kernel0 = 1;

    for (int i = 0; i < m_length; i++) {
        float k = dk * i;
        double x = beta * sqrt(1 - powf(2 * k / w, 2));

        data->at(i) = gsl_sf_bessel_I0(x) / w;

        if (i == 0) kernel0 = data->at(0);
        data->at(i) /= kernel0;

    }
    m_kernelData.reset(data);
}

ConvKernel::~ConvKernel()
{

}

const FloatVector *ConvKernel::getKernelData() const
{
    return m_kernelData.get();
}


float ConvKernel::getKernelWidth() const
{
    return m_kWidth;
}
