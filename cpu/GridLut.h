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

    void gridding2D(const ReconData &reconData, KData &out);

protected:
    ConvKernel m_kernel;
    int m_gridSize;
};

#endif // GRIDLUT_H
