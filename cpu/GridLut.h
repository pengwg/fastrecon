#ifndef GRIDLUT_H
#define GRIDLUT_H

#include "ConvKernel.h"

template<typename T>
class hostImageData;

template<typename T>
class cuImageData;

template<typename T>
class hostReconData;

template<typename T>
class cuReconData;

template<typename T>
class GridLut
{
public:
    GridLut(int gridSize, ConvKernel &kernel);
    virtual ~GridLut();

    cuImageData<T> gridding(cuReconData<T> &reconData);
    hostImageData<T> gridding(hostReconData<T> &reconData);

protected:
    int m_gridSize;
    ConvKernel m_kernel;
    FloatVector m_center[3];
    std::vector<int> m_start[3];
    std::vector<int> m_end[3];

    typename hostImageData<T>::LocalComplexVector *griddingChannel(const hostReconData<T> &reconData, int channel);
    typename cuImageData<T>::LocalComplexVector *griddingChannel(const cuReconData<T> &reconData, int channel);

    void cuPreprocess(const cuReconData<T> &reconData);
};

#endif // GRIDLUT_H
