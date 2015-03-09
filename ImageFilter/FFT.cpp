#include <string.h>
#include <iostream>
#include <QElapsedTimer>
#include <omp.h>

#include "FFT.h"

std::mutex FFT::m_mutex;

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

void FFT::plan()
{
    m_in.clear();
    m_plan.clear();

    std::cout << "Create CPU FFT plans for " << m_num_threads << " threads" << std::endl;

    // Create plans used by each threads
    std::lock_guard<std::mutex> lock(m_mutex);

    for (unsigned i = 0; i < m_num_threads; i++)
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

void FFT::excute(ImageData<float> &imgData)
{
    omp_set_num_threads(m_num_threads);

    if (m_plan.size() < m_num_threads)
    {
        plan();
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
            std::cout << "Thread " << id << " CPU FFT channel " << m_index << ':' << i << " | " << timer.restart() << " ms" << std::endl;
        }
    }
}

