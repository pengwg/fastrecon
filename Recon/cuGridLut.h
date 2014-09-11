#ifndef CUGRIDLUT_H
#define CUGRIDLUT_H

#include "GridLut.h"
#include "cuImageData.h"
#include "cuReconData.h"

typedef struct
{
    unsigned sample_idx;
    float delta;
} SampleTuple;

template<typename T>
class cuGridLut : public GridLut<T>
{
public:
    cuGridLut(int dim, int gridSize, ConvKernel &kernel);
    virtual ~cuGridLut() {}

    void plan(const cuVector<Point<T> > &traj);
    cuImageData<T> gridding(cuReconData<T> &reconData);

private:
    cuComplexVector<T> *griddingChannel(cuReconData<T> &reconData, int channel);

    std::unique_ptr<thrust::host_vector<SampleTuple>> m_tuples_last;
    std::unique_ptr<thrust::device_vector<unsigned>> m_cu_bucket_begin;
    std::unique_ptr<thrust::device_vector<unsigned>> m_cu_bucket_end;
};

#endif // CUGRIDLUT_H
