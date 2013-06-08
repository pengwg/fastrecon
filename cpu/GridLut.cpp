#include <QDebug>

#include "GridLut.h"


GridLut::GridLut(int gridSize, ConvKernel &kernel)
    : m_gridSize(gridSize), m_kernel(kernel)
{

}

GridLut::~GridLut()
{

}

template <typename T>
void GridLut::gridding(const ReconData<T> &reconData, KData &out)
{
    out.resize(powf(m_gridSize, reconData.rcDim()));

    float kHW = m_kernel.getKernelWidth() / 2;
    const QVector<float> *kernelData = m_kernel.getKernelData();
    int klength = kernelData->size();

    int idx = 0;
    const T *traj = reconData.getTraj();
    const KData *kData = reconData.getChannelData(0);

    float center[3] = {0};
    int start[3] = {0}, end[3] = {0};

    for (const auto &point : (*traj))
    {
        for (int i = 0; i < reconData.rcDim(); i++)
        {
            center[i] = (0.5 + point.pos[i]) * m_gridSize; // kspace in (-0.5, 0.5)
            start[i] = ceil(center[i] - kHW);
            end[i] = floor(center[i] + kHW);

            start[i] = fmax(start[i], 0);
            end[i] = fmin(end[i], m_gridSize - 1);
        }

        auto data = point.dcf * kData->at(idx);

        int i = start[2] * m_gridSize * m_gridSize + start[1] * m_gridSize + start[0];
        int di = m_gridSize - (end[0] - start[0]) - 1;
        int di2 = m_gridSize * m_gridSize - ((end[0] - start[0]) + 1) * ((end[1] - start[1]) + 1);

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
                        out[i] += kernelData->at(ki) * data;
                    }
                    i++;
                }
                i += di;
            }
            i += di2;
        }
        idx++;
    }
}

template
void GridLut::gridding(const ReconData<Traj2D> &reconData, KData &out);

template
void GridLut::gridding(const ReconData<Traj3D> &reconData, KData &out);
