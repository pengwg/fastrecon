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

void FFT::plan(int xSize, int ySize, bool forward)
{
    m_n0 = xSize;
    m_n1 = ySize;
    m_n2 = 1;
    
    m_in = (fftwf_complex *)fftwf_malloc(sizeof(fftw_complex) * xSize * ySize);
    int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;

    m_plan = fftwf_plan_dft_2d(ySize, xSize, m_in, m_in, sign, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
}

void FFT::plan(int xSize, int ySize, int zSize, bool forward)
{
    m_n0 = xSize;
    m_n1 = ySize;
    m_n2 = zSize;

    m_in = (fftwf_complex *)fftwf_malloc(sizeof(fftw_complex) * xSize * ySize * zSize);
    int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;

    m_plan = fftwf_plan_dft_3d(ySize, xSize, zSize, m_in, m_in, sign, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
}

void FFT::excute(ComplexVector &data)
{
    if (data.size() != m_n0 * m_n1 * m_n2)
    {
        std::cerr << "Error: FFT wrong data size" << std::endl;
        exit(1);
    }

    memcpy(m_in, data.data(), m_n0 * m_n1 * m_n2 * sizeof(fftwf_complex));
    fftwf_execute(m_plan);
    memcpy(data.data(), m_in, m_n0 * m_n1 * m_n2 * sizeof(fftwf_complex));
}

void FFT::fftShift(ComplexVector &data)
{
    if (data.size() != m_n0 * m_n1 * m_n2)
    {
        std::cerr << "Error: FFTSHIFT wrong data size" << std::endl;
        exit(1);
    }

    if (m_n2 > 1)
        fftShift3(data);
    else
        fftShift2(data);

}

void FFT::fftShift2(ComplexVector &data)
{
    int n0h = m_n0 / 2;
    int n1h = m_n1 / 2;

    int x1, y1;

    for (int y = 0; y < n1h; y++)
    {
        y1 = y + n1h;

        for (int x = 0; x < m_n0; x++)
        {
            x1 = x < n0h ? x + n0h : x - n0h;
            int i = y * m_n0 + x;
            int j = y1 * m_n0 + x1;

            std::swap(data[i], data[j]);
        }
    }
}

void FFT::fftShift3(ComplexVector &data)
{
    int n0h = m_n0 / 2;
    int n1h = m_n1 / 2;
    int n2h = m_n2 / 2;

    int x1, y1, z1;

    for (int z = 0; z < n2h; z++)
    {
        z1 = z + n2h;

        for (int y = 0; y < m_n1; y++)
        {
            y1 = y < n1h ? y + n1h : y - n1h;

            for (int x = 0; x < m_n0; x++)
            {
                x1 = x < n0h ? x + n0h : x - n0h;

                int i = z * m_n0 * m_n1 + y * m_n0 + x;
                int j = z1 * m_n0 * m_n1 + y1 * m_n0 + x1;

                std::swap(data[i], data[j]);
            }
        }
    }
}
