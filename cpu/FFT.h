#ifndef FFT_H
#define FFT_H

#include <fftw3.h>
#include "ReconData.h"

class FFT
{
public:
    FFT(int channels);
    ~FFT();

    void plan(int xSize, int ySize, bool forward);
    void plan(int xSize, int ySize, int zSize, bool forward);
    
    void excute(ImageData &imgData);
    void fftShift(ImageData &imgData);

private:
    int m_n0;
    int m_n1;
    int m_n2;
    int m_channels;

    std::vector<fftwf_plan> m_plan;
    std::vector<fftwf_complex *> m_in;

    void fftShift2(ComplexVector *data);
    void fftShift3(ComplexVector *data);
};

#endif // FFT_H
