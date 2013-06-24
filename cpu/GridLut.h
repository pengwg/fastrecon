#ifndef GRIDLUT_H
#define GRIDLUT_H

#include <complex>

#include "ConvKernel.h"
#include "ReconData.h"

class GridLut
{
public:
    GridLut(int gridSize, ConvKernel &kernel);
    virtual ~GridLut();

    ImageData gridding(ReconData &reconData);

protected:
    int m_gridSize;
    ConvKernel m_kernel;
    FloatVector m_center[3];
    std::vector<int> m_start[3];
    std::vector<int> m_end[3];

    ComplexVector *griddingChannel(const ReconData &reconData, int channel);
};

#endif // GRIDLUT_H
