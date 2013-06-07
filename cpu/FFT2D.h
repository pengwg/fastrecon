#ifndef FFT2D_H
#define FFT2D_H

#include <complex>
#include <fftw3.h>
#include <QVector>
#include "ReconData.h"


class FFT2D
{
public:
    FFT2D(int n0, int n1, bool forward);
    ~FFT2D();

    void excute(KData &data);
    void fftShift(KData &data);

private:
    const int m_n0;
    const int m_n1;

    fftwf_plan m_plan;
    fftwf_complex *m_in;
};

#endif // FFT2D_H
