#ifndef GRIDLUT_H
#define GRIDLUT_H

#include "ConvKernel.h"
#include "hostReconData.h"
#include "cuReconData.h"
#include "hostImageData.h"
#include "cuImageData.h"

template<typename T>
class GridLut
{
public:
    GridLut(int dim, int gridSize, ConvKernel &kernel);
    virtual ~GridLut();

    void cuPlan(const thrust::device_vector<Point<T> > &traj);

    cuImageData<T> gridding(cuReconData<T> &reconData);
    hostImageData<T> gridding(hostReconData<T> &reconData);

protected:
    typename hostImageData<T>::LocalComplexVector *griddingChannel(const hostReconData<T> &reconData, int channel);
    cuComplexVector<T> *griddingChannel(const cuReconData<T> &reconData, int channel);

    int m_dim;
    int m_gridSize;
    ConvKernel m_kernel;
    std::vector<float> m_center[3];
    std::vector<int> m_start[3];
    std::vector<int> m_end[3];

    std::unique_ptr<thrust::host_vector<int>> m_tuples_last;
    std::unique_ptr<thrust::device_vector<unsigned>> m_cu_bucket_begin;
    std::unique_ptr<thrust::device_vector<unsigned>> m_cu_bucket_end;
};

#endif // GRIDLUT_H
