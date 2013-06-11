#include <QDebug>

#include "GridLut.h"


GridLut::GridLut(int gridSize, ConvKernel &kernel)
    : m_gridSize(gridSize), m_kernel(kernel)
{

}

GridLut::~GridLut()
{

}

template <int N>
void GridLut::gridding(const ReconData<N> &reconData, ComplexVector &out)
{
    out.resize(powf(m_gridSize, N));

    float kHW = m_kernel.getKernelWidth() / 2;
    const std::vector<float> *kernelData = m_kernel.getKernelData();
    int klength = kernelData->size();

    const auto *traj = reconData.getTraj();
    const ComplexVector *kData = reconData.getChannelData(0);

    float center[3] = {0};
    int start[3] = {0}, end[3] = {0};
    auto itData = kData->cbegin();

    for (const auto &point : (*traj))
    {
        for (int i = 0; i < reconData.rcDim(); i++)
        {
            center[i] = (0.5 + point.pos[i]) * (m_gridSize - 1); // kspace in (-0.5, 0.5)
            start[i] = ceil(center[i] - kHW);
            end[i] = floor(center[i] + kHW);

            start[i] = fmax(start[i], 0);
            end[i] = fmin(end[i], m_gridSize - 1);
        }

        auto data = point.dcf * (*itData++);

        int i = start[2] * m_gridSize * m_gridSize + start[1] * m_gridSize + start[0];
        auto itOut = out.begin() + i;
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
                        int ki = round(dk / kHW * (klength - 1));
                        *itOut += kernelData->at(ki) * data;
                    }
                    itOut++;
                }
                itOut += di;
            }
            itOut += di2;
        }
    }
}

template
void GridLut::gridding(const ReconData<2> &reconData, ComplexVector &out);

template
void GridLut::gridding(const ReconData<3> &reconData, ComplexVector &out);
