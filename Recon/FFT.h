#ifndef FFT_H
#define FFT_H

#include <fftw3.h>
#include "ReconData.h"
#include "ImageData.h"

class FFT
{
public:
    FFT(int dims, ImageSize size, int sign = FFTW_BACKWARD);
    virtual ~FFT();

    virtual void plan();
    virtual void excute(ImageData<float> &imgData);

protected:
    int m_dim;
    ImageSize m_size;

private:
    int m_sign;
    std::vector<fftwf_plan> m_plan;
    std::vector<fftwf_complex *> m_in;
};

#endif // FFT_H
