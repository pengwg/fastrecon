#ifndef FFT_H
#define FFT_H

#include <fftw3.h>
#include "ReconData.h"

class FFT
{
public:
    FFT();
    ~FFT();

    void plan(int xSize, int ySize, bool forward);
    void plan(int xSize, int ySize, int zSize, bool forward);
    
    void excute(ComplexVector &data);
    void fftShift(ComplexVector &data);

private:
    int m_n0;
    int m_n1;
    int m_n2;

    fftwf_plan m_plan = nullptr;
    fftwf_complex *m_in = nullptr;

    void fftShift2(ComplexVector &data);
    void fftShift3(ComplexVector &data);
};

#endif // FFT_H
