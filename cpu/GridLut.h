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

    void gridding(const ReconData &reconData, ImageData &imgData);

protected:
    ConvKernel m_kernel;
    int m_gridSize;
    FloatVector m_center[3];
    std::vector<int> m_start[3];
    std::vector<int> m_end[3];

    void griddingChannel(const ReconData &reconData, int channel, ComplexVector &out);
};

#endif // GRIDLUT_H
