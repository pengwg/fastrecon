#include <string.h>
#include <iostream>
#include <QElapsedTimer>
#include <omp.h>

#include "FFT.h"

FFT::FFT(int channels)
    : m_channels(channels)
{

}

FFT::~FFT()
{
    for (auto plan : m_plan)
    {
        fftwf_destroy_plan(plan);
    }
    
    for (auto in : m_in)
    {
        fftwf_free(in);
    }
}

void FFT::plan(int xSize, int ySize, bool forward)
{
    m_n0 = xSize;
    m_n1 = ySize;
    m_n2 = 1;
    int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;

    m_in.clear();
    m_plan.clear();
    for (int i = 0; i < m_channels; i++)
    {
        auto in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * xSize * ySize);
        fftwf_plan plan = fftwf_plan_dft_2d(ySize, xSize, in, in, sign, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

        m_in.push_back(in);
        m_plan.push_back(plan);
    }
}

void FFT::plan(int xSize, int ySize, int zSize, bool forward)
{
    m_n0 = xSize;
    m_n1 = ySize;
    m_n2 = zSize;
    int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;

    m_in.clear();
    m_plan.clear();

    // Create plan for each channel used by multi-thread
    for (int i = 0; i < m_channels; i++)
    {
        auto in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * xSize * ySize * zSize);
        fftwf_plan plan = fftwf_plan_dft_3d(ySize, xSize, zSize, in, in, sign, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

        m_in.push_back(in);
        m_plan.push_back(plan);
    }
}

void FFT::excute(ImageData &imgData)
{
#pragma omp parallel shared(imgData)
    {
        int id = omp_get_thread_num();
        QElapsedTimer timer;
        timer.start();

#pragma omp for schedule(dynamic)
        for (int i = 0; i < m_channels; i++)
        {
            auto data = imgData[i].get();
            if (data->size() != m_n0 * m_n1 * m_n2)
            {
                std::cerr << "Error: FFT wrong data size" << std::endl;
                exit(1);
            }

            auto in = m_in[i];
            auto plan = m_plan[i];
            memcpy(in, data->data(), m_n0 * m_n1 * m_n2 * sizeof(fftwf_complex));
            fftwf_execute(plan);
            memcpy(data->data(), in, m_n0 * m_n1 * m_n2 * sizeof(fftwf_complex));

#pragma omp critical
            std::cout << "Thread " << id << " FFT channel " << i << " | " << timer.restart() << " ms" << std::endl;
        }
    }
}

void FFT::fftShift(ImageData &imgData)
{
    for (int n = 0; n < m_channels; n++)
    {
        auto data = imgData[n].get();
        if (data->size() != m_n0 * m_n1 * m_n2)
        {
            std::cerr << "Error: FFTSHIFT wrong data size" << std::endl;
            exit(1);
        }

        if (m_n2 > 1)
            fftShift3(data);
        else
            fftShift2(data);
    }
}

void FFT::fftShift2(ComplexVector *data)
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

            std::swap(data->at(i), data->at(j));
        }
    }
}

void FFT::fftShift3(ComplexVector *data)
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

                std::swap(data->at(i), data->at(j));
            }
        }
    }
}
