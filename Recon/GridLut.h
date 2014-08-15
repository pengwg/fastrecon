#ifndef GRIDLUT_H
#define GRIDLUT_H

#include "ConvKernel.h"
#include "ReconData.h"
#include "cuReconData.h"
#include "ImageData.h"

template<typename T>
class GridLut
{
public:
    GridLut(int dim, int gridSize, ConvKernel &kernel);
    virtual ~GridLut();

    void cuPlan(const cuVector<Point<T> > &traj);

    ImageData<T> gridding(cuReconData<T> &reconData);
    ImageData<T> gridding(ReconData<T> &reconData);

protected:
    ComplexVector<T> *griddingChannel(const ReconData<T> &reconData, int channel);
    cuComplexVector<T> *griddingChannel(cuReconData<T> &reconData, int channel);

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
