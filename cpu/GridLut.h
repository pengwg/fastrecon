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

class GridLut
{
public:
    GridLut(int gridSize, ConvKernel &kernel);
    virtual ~GridLut();

    template<typename T>
    cuImageData<T> gridding(cuReconData<T> &reconData);
    template<typename T>
    hostImageData<T> gridding(hostReconData<T> &reconData);

protected:
    int m_gridSize;
    ConvKernel m_kernel;
    FloatVector m_center[3];
    std::vector<int> m_start[3];
    std::vector<int> m_end[3];

    template<typename T>
    typename hostImageData<T>::LocalComplexVector *griddingChannel(const hostReconData<T> &reconData, int channel);
    template<typename T>
    typename cuImageData<T>::LocalComplexVector *griddingChannel(const cuReconData<T> &reconData, int channel);

    template<typename T>
    void cuPreprocess(cuReconData<T> &reconData);
};

#endif // GRIDLUT_H
