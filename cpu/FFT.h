#ifndef FFT_H
#define FFT_H

#include <fftw3.h>
#include "ReconData.h"

class FFT
{
public:
    FFT();
    ~FFT();

    void plan(int n0, int n1, bool forward);
    void plan(int n0, int n1, int n2, bool forward);
    
    void excute(KData &data);
    void fftShift(KData &data);

private:
    int m_n0 = 0;
    int m_n1 = 1;
    int m_n2 = 1;

    fftwf_plan m_plan = nullptr;
    fftwf_complex *m_in = nullptr;
};

#endif // FFT_H
