#ifndef FFT_H
#define FFT_H

#include <fftw3.h>
#include "ReconData.h"
#include "ImageData.h"

class FFT
{
public:
    FFT(int dims, ImageSize size, int sign = FFTW_BACKWARD);
    ~FFT();

    void plan(int threads);
    
    void excute(ImageData<std::vector, float> &imgData);
    void fftShift(ImageData<std::vector, float> &imgData);

private:
    int m_dim;
    ImageSize m_size;
    bool m_sign;

    std::vector<fftwf_plan> m_plan;
    std::vector<fftwf_complex *> m_in;
};

#endif // FFT_H
