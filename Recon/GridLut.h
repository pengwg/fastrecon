#ifndef GRIDLUT_H
#define GRIDLUT_H

#include "ConvKernel.h"
#include "ReconData.h"
#include "ImageData.h"


template<typename T>
class GridLut
{
public:
    GridLut(int dim, int gridSize, ConvKernel &kernel);
    virtual ~GridLut();

    ImageData<T> gridding(ReconData<T> &reconData);

protected:
    ComplexVector<T> *griddingChannel(const ReconData<T> &reconData, int channel);

    int m_dim;
    int m_gridSize;
    ConvKernel m_kernel;
    std::vector<float> m_center[3];
    std::vector<int> m_start[3];
    std::vector<int> m_end[3];
};

#endif // GRIDLUT_H
