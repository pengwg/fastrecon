#include <QElapsedTimer>
#include <iostream>
#include <omp.h>

#include "GridLut.h"
#ifdef BUILD_CUDA
#include "cuGridLut.h"
#endif // BUILD_CUDA


template<typename T>
std::shared_ptr<GridLut<T>> GridLut<T>::Create(ReconData<T> &reconData, unsigned gridSize, const ConvKernel &kernel)
{
    GridLut<T> *instance = nullptr;

#ifdef BUILD_CUDA
    auto cu_data = dynamic_cast<cuReconData<T> *>(&reconData);
    if (cu_data != nullptr)
        instance = new cuGridLut<T>(*cu_data, kernel);
#endif // BUILD_CUDA
    if (instance == nullptr)
        instance = new GridLut<T>(reconData, kernel);

    instance->m_dim = reconData.rcDim();
    instance->m_gridSize = gridSize;

    return std::shared_ptr<GridLut<T>>(instance);;
}

template<typename T>
GridLut<T>::~GridLut()
{

}

template<typename T>
void GridLut<T>::plan()
{
    m_associatedData.normalizeTraj(m_gridSize);
}

template<typename T>
std::shared_ptr<ImageData<T> > GridLut<T>::execute()
{
    omp_set_num_threads(m_num_threads);

    auto img = new ImageData<T>(m_dim, {m_gridSize, m_gridSize, m_gridSize});

#pragma omp parallel shared(img)
    {
        int id = omp_get_thread_num();
        QElapsedTimer timer;
        timer.start();
#pragma omp for schedule(dynamic) ordered
        for (int i = 0; i < m_associatedData.channels(); i++)
        {
            auto out = griddingChannel(i);
#pragma omp ordered
            {
                img->addChannelImage(std::move(out));
                std::cout << "Thread " << id << " cpu gridding channel " << m_index << ':' << i << " | " << timer.restart() << " ms" << std::endl;
            }
        }
    }
    return std::shared_ptr<ImageData<T>>(img);
}

template<typename T>
std::unique_ptr<ComplexVector<T>> GridLut<T>::griddingChannel(int channel)
{
    const ComplexVector<T> *kData = m_associatedData.getChannelData(channel);
    auto itDcf = m_associatedData.getDcf().cbegin();

    float kHW = m_kernel.getKernelWidth() / 2;
    const std::vector<float> *kernelData = m_kernel.getKernelData();
    int klength = kernelData->size();

    auto itTraj = m_associatedData.getTraj().cbegin();

    float center[3] = {0};
    int start[3] = {0}, end[3] = {0};
    auto out = new ComplexVector<T>(std::pow(m_gridSize, m_dim));

    for (const auto &sample : (*kData))
    {
        for (auto i = 0u; i < m_dim; i++)
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
                        int ki = (int)std::round(dk / kHW * (klength - 1));
                        *itOut += kernelData->at(ki) * data;
                    }
                    itOut++;
                }
                itOut += di;
            }
            itOut += di2;
        }
    }
    return std::unique_ptr<ComplexVector<T>>(out);
}

template class GridLut<float>;
