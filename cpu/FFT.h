#ifndef FFT_H
#define FFT_H

#include <fftw3.h>
#include "ReconData.h"

class FFT
{
public:
    FFT(int dims, int channels, int size, int sign = FFTW_BACKWARD);
    ~FFT();

    void plan(int threads, int xSize, int ySize, int zSize = 1);
    
    void excute(ImageData &imgData);
    void fftShift(ImageData &imgData);

private:
    int m_n0;
    int m_n1;
    int m_n2;
    int m_dims;
    int m_channels;
    int m_size;
    bool m_sign;

    std::vector<fftwf_plan> m_plan;
    std::vector<fftwf_complex *> m_in;

    void fftShift2(ComplexVector *data);
    void fftShift3(ComplexVector *data);
};

#endif // FFT_H
