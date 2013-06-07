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

    // void gridding(const ReconData<Traj2D> &reconData, KData &out);

    template <typename T>
    void gridding(const ReconData<T> &reconData, KData &out);

protected:
    ConvKernel m_kernel;
    int m_gridSize;
};

#endif // GRIDLUT_H
