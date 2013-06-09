#include <string.h>
#include <iostream>

#include "FFT.h"

FFT::FFT()
{
}

FFT::~FFT()
{
    if (m_plan)
        fftwf_destroy_plan(m_plan);
    
    if (m_in)
        fftwf_free(m_in);
}

void FFT::plan(int n0, int n1, bool forward)
{
    m_n0 = n0;
    m_n1 = n1;
    
    m_in = (fftwf_complex *)fftwf_malloc(sizeof(fftw_complex) * n0 * n1);
    int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;

    m_plan = fftwf_plan_dft_2d(n0, n1, m_in, m_in, sign, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
}

void FFT::plan(int n0, int n1, int n2, bool forward)
{
    m_n0 = n0;
    m_n1 = n1;
    m_n2 = n2;

    m_in = (fftwf_complex *)fftwf_malloc(sizeof(fftw_complex) * n0 * n1 * n2);
    int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;

    m_plan = fftwf_plan_dft_3d(n0, n1, n2, m_in, m_in, sign, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
}

void FFT::excute(KData &data)
{
    if (data.size() != m_n0 * m_n1 * m_n2)
    {
        std::cerr << "Error: wrong FFT size" << std::endl;
        exit(1);
    }

    memcpy(m_in, data.data(), m_n0 * m_n1 * m_n2 * sizeof(fftwf_complex));
    fftwf_execute(m_plan);
    memcpy(data.data(), m_in, m_n0 * m_n1 * m_n2 * sizeof(fftwf_complex));
}

void FFT::fftShift(KData &data)
{
    int n0h = m_n0 / 2;
    int n1h = m_n1 / 2;

    int x1, y1;

    for (int y = 0; y < n0h; y++) {
        y1 = y + n0h;

        for (int x = 0; x < m_n1; x++) {
            if (x < n1h)
                x1 = x + n1h;
            else
                x1 = x - n1h;

            int i = y * m_n1 + x;
            int j = y1 * m_n1 + x1;

            auto tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}
