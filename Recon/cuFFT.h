#ifndef CUFFT_H
#define CUFFT_H

#include <cufft.h>
#include "cuImageData.h"

class cuFFT
{
public:
    cuFFT(int dims, ImageSize size, int sign = CUFFT_INVERSE);
    ~cuFFT();

    void plan();
    void excute(cuImageData<float> &imgData);

private:
    int m_dim;
    ImageSize m_size;
    int m_sign;

    cufftHandle m_plan;
};

#endif // CUFFT_H
