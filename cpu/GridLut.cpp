#include <QDebug>

#include "GridLut.h"


GridLut::GridLut(int gridSize, ConvKernel &kernel)
    : m_gridSize(gridSize), m_kernel(kernel)
{

}

GridLut::~GridLut()
{

}

void GridLut::gridding(const ReconData<Traj2D> &reconData, KData &out)
{
    out.resize(m_gridSize * m_gridSize);

    float kHW = m_kernel.getKernelWidth() / 2;
    const QVector<float> *kernelData = m_kernel.getKernelData();
    int klength = kernelData->size();

    int idx = 0;
    const Traj2D *traj = reconData.getTraj();
    const KData *kData = reconData.getChannelData(0);

    for (const KPoint2D &point : (*traj)) {
        float xCenter = (0.5 + point.kx) * m_gridSize; // kx in (-0.5, 0.5)
        int xStart = ceil(xCenter - kHW);
        int xEnd = floor(xCenter + kHW);

        float yCenter = (0.5 + point.ky) * m_gridSize; // ky in (-0.5, 0.5)
        int yStart = ceil(yCenter  - kHW);
        int yEnd = floor(yCenter + kHW);

        xStart = fmax(xStart, 0);
        xEnd = fmin(xEnd, m_gridSize - 1);

        yStart = fmax(yStart, 0);
        yEnd = fmin(yEnd, m_gridSize - 1);

        auto data = point.dcf * kData->at(idx);

        int i = yStart * m_gridSize + xStart;
        int di = m_gridSize - (xEnd - xStart) - 1;

        for (int y = yStart; y <= yEnd; y++) {
            float dy = y - yCenter;
            float dy2 = dy * dy;

            for (int x = xStart; x <= xEnd; x++) {
                float dx = x - xCenter;
                float dk = sqrtf(dy2 + dx * dx);

                if (dk < kHW) {
                    int ki = round(dk / kHW * (klength - 1));
                    out[i] += kernelData->at(ki) * data;
                }
                i++;
            }
            i += di;
        }
        idx++;
    }
}

void GridLut::gridding(const ReconData<Traj3D> &reconData, KData &out)
{

}
