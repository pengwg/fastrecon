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

typedef struct
{
    thrust::host_vector<unsigned> bucket_begin;
    thrust::host_vector<unsigned> bucket_end;
    thrust::host_vector<SampleTuple> tuples_last;
} DataMap;

typedef struct
{
    thrust::device_vector<unsigned> bucket_begin;
    thrust::device_vector<unsigned> bucket_end;
    thrust::device_vector<SampleTuple> tuples_last;
} cuDataMap;

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
    void addDataMapFromDevice();

    std::unique_ptr<thrust::host_vector<SampleTuple>> m_tuples_last;
    std::unique_ptr<thrust::device_vector<unsigned>> m_cu_bucket_begin;
    std::unique_ptr<thrust::device_vector<unsigned>> m_cu_bucket_end;

    std::vector<DataMap> m_all_data_map;
    cuDataMap m_cu_data_map;
    int m_index_data_map_in_device = -1;
};

#endif // CUGRIDLUT_H
