#include <string.h>
#include <iostream>
#include <QElapsedTimer>
#include <omp.h>

#include "FFT.h"

FFT::FFT(int dims, ImageSize size, int sign)
    : m_dims(dims), m_size(size), m_sign(sign)
{
    if (dims == 2)
        m_size.z = 1;
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

void FFT::plan(int threads)
{
    m_in.clear();
    m_plan.clear();

    // Create plans used by each threads
    for (int i = 0; i < threads; i++)
    {
        auto in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * m_size.x * m_size.y * m_size.z);
        fftwf_plan plan;
        if (m_dims == 2)
            plan = fftwf_plan_dft_2d(m_size.y, m_size.x, in, in, m_sign, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
        else if (m_dims ==3)
            plan = fftwf_plan_dft_3d(m_size.z, m_size.y, m_size.x, in, in, m_sign, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

        m_in.push_back(in);
        m_plan.push_back(plan);
    }
}

void FFT::excute(ImageData &imgData)
{
    int threads = omp_get_max_threads();
    if (m_plan.size() < threads)
    {
        std::cout << "Create plans for " << threads << " threads" << std::endl;
        plan(threads);
    }
    if (imgData.length() != m_size.x * m_size.y * m_size.z)
    {
        std::cerr << "Error: FFT wrong image size" << std::endl;
        exit(1);
    }

#pragma omp parallel shared(imgData)
    {
        int id = omp_get_thread_num();
        QElapsedTimer timer;
        timer.start();

#pragma omp for schedule(dynamic)
        for (int i = 0; i < imgData.channels(); i++)
        {
            auto data = imgData.getChannelImage(i);
            auto in = m_in[id];
            auto plan = m_plan[id];

            memcpy(in, data->data(), imgData.length() * sizeof(fftwf_complex));
            fftwf_execute(plan);
            memcpy(data->data(), in, imgData.length() * sizeof(fftwf_complex));

#pragma omp critical
            std::cout << "Thread " << id << " FFT channel " << i << " | " << timer.restart() << " ms" << std::endl;
        }
    }
}

void FFT::fftShift(ImageData &imgData)
{
    if (imgData.length() != m_size.x * m_size.y * m_size.z)
    {
        std::cerr << "Error: FFTSHIFT wrong image size" << std::endl;
        exit(1);
    }

    for (int n = 0; n < imgData.channels(); n++)
    {
        auto data = imgData.getChannelImage(n);

        if (m_dims == 3)
            fftShift3(data);
        else
            fftShift2(data);
    }
}

void FFT::fftShift2(ComplexVector *data)
{
    int n0h = m_size.x / 2;
    int n1h = m_size.y / 2;

    int x1, y1;

    for (int y = 0; y < n1h; y++)
    {
        y1 = y + n1h;

        for (int x = 0; x < m_size.x; x++)
        {
            x1 = x < n0h ? x + n0h : x - n0h;
            int i = y * m_size.x + x;
            int j = y1 * m_size.x + x1;

            std::swap(data->at(i), data->at(j));
        }
    }
}

void FFT::fftShift3(ComplexVector *data)
{
    int n0h = m_size.x / 2;
    int n1h = m_size.y / 2;
    int n2h = m_size.z / 2;

    int x1, y1, z1;

    for (int z = 0; z < n2h; z++)
    {
        z1 = z + n2h;

        for (int y = 0; y < m_size.y; y++)
        {
            y1 = y < n1h ? y + n1h : y - n1h;

            for (int x = 0; x < m_size.x; x++)
            {
                x1 = x < n0h ? x + n0h : x - n0h;

                int i = z * m_size.x * m_size.y + y * m_size.x + x;
                int j = z1 * m_size.x * m_size.y + y1 * m_size.x + x1;

                std::swap(data->at(i), data->at(j));
            }
        }
    }
}
