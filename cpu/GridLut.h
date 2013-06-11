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

    template <int N>
    void gridding(const ReconData<N> &reconData, ComplexVector &out);

protected:
    ConvKernel m_kernel;
    int m_gridSize;
};

#endif // GRIDLUT_H
