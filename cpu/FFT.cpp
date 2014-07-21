#include <string.h>
#include <iostream>
#include <QElapsedTimer>
#include <omp.h>

#include "FFT.h"

FFT::FFT(int dims, ImageSize size, int sign)
    : m_dim(dims), m_size(size), m_sign(sign)
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
        if (m_dim == 2)
            plan = fftwf_plan_dft_2d(m_size.y, m_size.x, in, in, m_sign, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
        else if (m_dim ==3)
            plan = fftwf_plan_dft_3d(m_size.z, m_size.y, m_size.x, in, in, m_sign, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);

        m_in.push_back(in);
        m_plan.push_back(plan);
    }
}

void FFT::excute(ImageData<std::vector, float> &imgData)
{
    int threads = omp_get_max_threads();
    if (m_plan.size() < threads)
    {
        std::cout << "Create plans for " << threads << " threads" << std::endl;
        plan(threads);
    }
    if (imgData.dataSize() != m_size.x * m_size.y * m_size.z)
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

            memcpy(in, data->data(), imgData.dataSize() * sizeof(fftwf_complex));
            fftwf_execute(plan);
            memcpy(data->data(), in, imgData.dataSize() * sizeof(fftwf_complex));

#pragma omp critical
            std::cout << "Thread " << id << " FFT channel " << i << " | " << timer.restart() << " ms" << std::endl;
        }
    }
}

