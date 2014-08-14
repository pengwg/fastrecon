#include <QElapsedTimer>
#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>

#include "GridLut.h"

template<typename T>
GridLut<T>::GridLut(int dim, int gridSize, ConvKernel &kernel)
    : m_dim(dim), m_gridSize(gridSize), m_kernel(kernel)
{

}

template<typename T>
GridLut<T>::~GridLut()
{

}

// ----------CUDA Testing-----------------
template<typename T>
ImageData<T> GridLut<T>::gridding(cuReconData<T> &reconData)
{
    auto bounds = reconData.getCompBounds(0);
    auto tr = -bounds.first;
    auto scale = (m_gridSize - 1) / (bounds.second - bounds.first);

    reconData.transformTraj(tr, scale);

    QElapsedTimer timer;
    timer.start();

    //cuPlan(*reconData.getTraj());

    cudaDeviceSynchronize();

    std::cout << "GPU preprocess " << " | " << timer.restart() << " ms" << std::endl;

    ImageData<T> img(reconData.rcDim(), {m_gridSize, m_gridSize, m_gridSize});

    for (int i = 0; i < reconData.channels(); i++)
    {
        //auto out = griddingChannel(reconData, i);
        //img.addChannelImage(out);
        std::cout << "GPU gridding channel " << i << " | " << timer.restart() << " ms" << std::endl;
    }
    return img;
}

template<typename T>
ImageData<T> GridLut<T>::gridding(hostReconData<T> &reconData)
{
    auto bounds = reconData.getCompBounds(0);
    auto tr = -bounds.first;
    auto scale = (m_gridSize - 1) / (bounds.second - bounds.first);

    reconData.transformTraj(tr, scale);

    ImageData<T> img(reconData.rcDim(), {m_gridSize, m_gridSize, m_gridSize});

#pragma omp parallel shared(img, reconData)
    {
        int id = omp_get_thread_num();
        QElapsedTimer timer;
        timer.start();
#pragma omp for schedule(dynamic) ordered
        for (int i = 0; i < reconData.channels(); i++)
        {
            auto out = griddingChannel(reconData, i);
#pragma omp ordered
            {
                img.addChannelImage(out);
                std::cout << "Thread " << id << " gridding channel " << i << " | " << timer.restart() << " ms" << std::endl;
            }
        }
    }
    return img;
}

template<typename T>
ComplexVector<T> *GridLut<T>::griddingChannel(const hostReconData<T> &reconData, int channel)
{
    const ComplexVector<T> *kData = reconData.getChannelData(channel);
    auto itDcf = reconData.getDcf()->cbegin();

    float kHW = m_kernel.getKernelWidth() / 2;
    const std::vector<float> *kernelData = m_kernel.getKernelData();
    int klength = kernelData->size();

    auto itTraj = reconData.getTraj()->cbegin();

    float center[3] = {0};
    int start[3] = {0}, end[3] = {0};
    ComplexVector<T> *out = new ComplexVector<T>(powf(m_gridSize, m_dim));

    for (const auto &sample : (*kData))
    {
        for (int i = 0; i < m_dim; i++)
        {
            center[i] = itTraj->x[i]; //(0.5 + *itTrajComp[i]++) * (m_gridSize - 1); // kspace in (-0.5, 0.5)
            start[i] = ceil(center[i] - kHW);
            end[i] = floor(center[i] + kHW);

            start[i] = fmax(start[i], 0);
            end[i] = fmin(end[i], m_gridSize - 1);
        }
        ++itTraj;

        auto data = (*itDcf++) * sample;

        int i = start[2] * m_gridSize * m_gridSize + start[1] * m_gridSize + start[0];
        auto itOut = out->begin() + i;

        // Step size for linear addressing
        int di = m_gridSize - (end[0] - start[0]) - 1;
        int di2 = m_gridSize * m_gridSize - (end[1] - start[1] + 1) * m_gridSize;

        for (int z = start[2]; z <= end[2]; z++)
        {
            float dz = z - center[2];
            float dz2 = dz * dz;

            for (int y = start[1]; y <= end[1]; y++)
            {
                float dy = y - center[1];
                float dy2 = dy * dy;

                for (int x = start[0]; x <= end[0]; x++)
                {
                    float dx = x - center[0];
                    float dk = sqrtf(dz2 + dy2 + dx * dx);

                    if (dk < kHW)
                    {
                        // kernelData has hight resolution, interpolation error can be ignored
                        int ki = (int)(dk / kHW * (klength - 1));
                        *itOut += kernelData->at(ki) * data;
                    }
                    itOut++;
                }
                itOut += di;
            }
            itOut += di2;
        }
    }
    return out;
}

template class GridLut<float>;