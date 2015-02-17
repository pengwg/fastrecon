#ifndef CUGRIDLUT_H
#define CUGRIDLUT_H

#include <mutex>

#include "GridLut.h"
#include "cuReconData.h"
#include "cuImageData.h"

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
    cuGridLut(unsigned dim, unsigned gridSize, ConvKernel &kernel);
    virtual ~cuGridLut() {}

    virtual void plan(ReconData<T> &reconData) override;
    virtual std::unique_ptr<ImageData<T>> execute(ReconData<T> &reconData) override;
    void setNumOfPartitions(int partitions) {
        m_gpu_partitions = partitions;
    }

private:
    std::unique_ptr<cuComplexVector<T>> griddingChannel(cuReconData<T> &reconData, int channel);
    void addDataMapFromDevice();
    const cuDataMap *getDeviceDataMapPartition(int index);

    static std::vector<DataMap> m_all_data_map;
    cuDataMap m_cu_data_map;
    int m_index_data_map_in_device = -1;
    int m_gpu_partitions = 1;

    static std::mutex m_mutex;
    static const cuDataMap *m_cu_data_map_persistent;
};

#endif // CUGRIDLUT_H
