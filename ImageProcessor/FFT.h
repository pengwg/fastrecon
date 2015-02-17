#ifndef FFT_H
#define FFT_H

#include <fftw3.h>
#include <mutex>

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
    void setIndex(int index) {
        m_index = index;
    }

protected:
    int m_dim;
    ImageSize m_size;
    int m_index = 0;

private:
    int m_sign;
    std::vector<fftwf_plan> m_plan;
    std::vector<fftwf_complex *> m_in;

    unsigned m_num_threads = 1; // number of child threads for openmp
    static std::mutex m_mutex;
};

#endif // FFT_H
