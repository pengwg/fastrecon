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
    void setNumOfThreads(unsigned threads) {
        m_num_threads = threads;
    }

protected:
    int m_dim;
    ImageSize m_size;

private:
    int m_sign;
    std::vector<fftwf_plan> m_plan;
    std::vector<fftwf_complex *> m_in;

    unsigned m_num_threads = 1;
};

#endif // FFT_H
