#include <QDebug>

#include "GridLut.h"


GridLut::GridLut(int gridSize, ConvKernel &kernel)
    : m_gridSize(gridSize), m_kernel(kernel)
{

}

GridLut::~GridLut()
{

}

void GridLut::gridding(const ReconData &reconData, ImageData &imgData)
{
    int rcDim = reconData.rcDim();
    for (int i = 0; i < reconData.channels(); i++)
    {
        ComplexVector *out = new ComplexVector(powf(m_gridSize, rcDim));
        griddingChannel(reconData, i, *out);
        imgData.push_back(std::shared_ptr<ComplexVector>(out));
    }
}

void GridLut::griddingChannel(const ReconData &reconData, int channel, ComplexVector &out)
{
    const ComplexVector *kData = reconData.getChannelData(channel);
    auto itDcf = reconData.getDcf()->cbegin();
    int rcDim = reconData.rcDim();

    float kHW = m_kernel.getKernelWidth() / 2;
    const std::vector<float> *kernelData = m_kernel.getKernelData();
    int klength = kernelData->size();

    FloatVector::const_iterator itTrajComp[3];
    for (int i = 0; i < rcDim; i++)
    {
        itTrajComp[i] = reconData.getTrajComponent(i)->cbegin();
    }

    float center[3] = {0};
    int start[3] = {0}, end[3] = {0};

    for (const auto &sample : (*kData))
    {
        for (int i = 0; i < rcDim; i++)
        {
            center[i] = (0.5 + *itTrajComp[i]++) * (m_gridSize - 1); // kspace in (-0.5, 0.5)
            start[i] = ceil(center[i] - kHW);
            end[i] = floor(center[i] + kHW);

            start[i] = fmax(start[i], 0);
            end[i] = fmin(end[i], m_gridSize - 1);
        }

        auto data = (*itDcf++) * sample;

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

