#ifndef CUFFT_H
#define CUFFT_H

#include <cufft.h>
#include "ImageData.h"
#include "FFT.h"

class cuFFT : public FFT
{
public:
    cuFFT(int dims, ImageSize size, int sign = CUFFT_INVERSE);
    virtual ~cuFFT();

    virtual void plan() override;
    virtual void excute(ImageData<float> &imgData) override;

private:
    int m_sign;
    cufftHandle m_plan = 0;
};

#endif // CUFFT_H
