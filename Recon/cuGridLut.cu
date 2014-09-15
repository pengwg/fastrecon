#include <QElapsedTimer>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/system/omp/execution_policy.h>
#include <cassert>

#include "cuGridLut.h"

template<typename T>
cuGridLut<T>::cuGridLut(int dim, int gridSize, ConvKernel &kernel)
    : GridLut<T>(dim, gridSize, kernel)
{
}

template<typename T>
cuImageData<T> cuGridLut<T>::gridding(cuReconData<T> &reconData)
{
    QElapsedTimer timer;
    timer.start();

    auto bounds = reconData.getCompBounds(0);
    auto tr = -bounds.first;
    auto scale = (this->m_gridSize - 1) / (bounds.second - bounds.first);

    reconData.transformTraj(tr, scale);

    plan(*reconData.cuGetTraj());

    std::cout << "GPU preprocess " << " | " << timer.restart() << " ms" << std::endl;

    cuImageData<T> img(reconData.rcDim(), {this->m_gridSize, this->m_gridSize, this->m_gridSize});

    for (int i = 0; i < reconData.channels(); i++)
    {
        auto out = griddingChannel(reconData, i);
        img.addChannelImage(out);
        std::cout << "GPU gridding channel " << i << " | " << timer.restart() << " ms" << std::endl;
    }
    return img;
}

template<typename T>
struct compute_num_cells_per_sample
{
    T _half_W;
    int _dim;
    __host__ __device__
    compute_num_cells_per_sample(T half_W, int dim) : _half_W(half_W), _dim(dim) {}

    __host__ __device__
    unsigned operator()(const Point<T> &p) const
    {
        unsigned num_cells = 1;
        for( int d = 0; d < _dim; ++d) {
            unsigned upper_limit = (unsigned)floor((float)p.x[d] + _half_W);
            unsigned lower_limit = (unsigned)ceil((float)p.x[d] - _half_W);
            num_cells *= upper_limit - lower_limit + 1;
        }
        return num_cells;
    }
};

template<typename T> __global__
void write_pairs_kernel(const Point<T> *traj, unsigned *tuple_index, int *tuples_first, SampleTuple *tuples_last,
                        int reconSize, T half_W, size_t num_samples, size_t skip)
{
    size_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx >= num_samples)
        return;

    const Point<T> &sample = traj[sample_idx];
    int lb[3], ub[3];

    for (int i = 0; i < 3; ++i)
    {
        lb[i] = (int)ceil(sample.x[i] - half_W);
        ub[i] = (int)floor(sample.x[i] + half_W);
    }

    unsigned write_offset = tuple_index[sample_idx];
    int counter = 0;

    //if (sample_idx < 10)
        //printf("sample: %lu, offset: %u, lb: %d, ub: %d\n", sample_idx, write_offset, lb[2], ub[2]);

    for (int z = lb[2]; z <= ub[2]; ++z)
    {
        auto dz = z - sample.x[2];
        for (int y = lb[1]; y <= ub[1]; ++y)
        {
            auto dy = y - sample.x[1];
            for (int x = lb[0]; x <= ub[0]; ++x)
            {
                int matrix_index = x + y * reconSize + z * reconSize * reconSize;
                tuples_first[write_offset + counter] = matrix_index;
                tuples_last[write_offset + counter].sample_idx = sample_idx + skip;

                auto dx = x - sample.x[0];
                tuples_last[write_offset + counter].delta = sqrtf(dx * dx + dy * dy + dz * dz);
                ++counter;
            }
        }
    }
}

template<typename T>
void cuGridLut<T>::addDataMapFromDevice()
{
    m_all_data_map.emplace_back();
    auto &data_map = m_all_data_map.back();

    data_map.bucket_begin = m_cu_data_map.bucket_begin;
    data_map.bucket_end = m_cu_data_map.bucket_end;
    data_map.tuples_last = m_cu_data_map.tuples_last;

    m_index_data_map_in_device = m_all_data_map.size();
}

template<typename T>
void cuGridLut<T>::plan(const cuVector<Point<T>> &traj)
{
    T half_W = this->m_kernel.getKernelWidth() / 2.0;

    auto cells_per_sample = new thrust::device_vector<unsigned> (traj.size());
    thrust::transform(traj.begin(), traj.end(), cells_per_sample->begin(), compute_num_cells_per_sample<T>(half_W, this->m_dim));

    thrust::device_vector<unsigned> tuple_index(traj.size());
    thrust::inclusive_scan(cells_per_sample->begin(), cells_per_sample->end(), tuple_index.begin(), thrust::plus<unsigned> ());
    delete cells_per_sample;

    m_cu_data_map.bucket_begin.resize(powf(this->m_gridSize, this->m_dim));
    m_cu_data_map.bucket_end.resize(powf(this->m_gridSize, this->m_dim));
    thrust::device_vector<int> cu_tuples_first;

    std::cout << " Traj size: " << traj.size() << ", Image size: " << m_cu_data_map.bucket_begin.size() << ", Number of pairs: " << tuple_index.back() << std::endl;

    int num_partitions = 20;
    size_t chunk_size = ceil((double)traj.size() / num_partitions);

    size_t blockSize = 256;
    size_t gridSize = (size_t)ceil((double)chunk_size / blockSize);

    size_t skip = 0;
    while (skip < traj.size())
    {
        size_t num_of_samples_compute = std::min(chunk_size, traj.size() - skip);
        unsigned num_of_pairs = tuple_index[skip + num_of_samples_compute - 1] - tuple_index[skip];

        std::cout << "num_of_pairs compute: " << num_of_pairs << std::endl;

        cu_tuples_first.resize(num_of_pairs);
        m_cu_data_map.tuples_last.resize(num_of_pairs);

        const Point<T> *traj_ptr = thrust::raw_pointer_cast(traj.data()) + skip;
        unsigned *tuple_index_ptr = thrust::raw_pointer_cast(tuple_index.data()) + skip;
        int *tuples_first_ptr = thrust::raw_pointer_cast(cu_tuples_first.data()) - tuple_index[skip];
        SampleTuple *tuples_last_ptr = thrust::raw_pointer_cast(m_cu_data_map.tuples_last.data()) - tuple_index[skip];

        write_pairs_kernel<T><<<gridSize, blockSize>>>(traj_ptr, tuple_index_ptr, tuples_first_ptr, tuples_last_ptr,
                                                       this->m_gridSize, half_W, num_of_samples_compute, skip);

        thrust::sort_by_key(cu_tuples_first.begin(), cu_tuples_first.end(), m_cu_data_map.tuples_last.begin());


        thrust::counting_iterator<int> search_begin(0);
        thrust::lower_bound(cu_tuples_first.begin(), cu_tuples_first.end(), search_begin,
                            search_begin + (int)powf(this->m_gridSize, this->m_dim), m_cu_data_map.bucket_begin.begin());

        thrust::upper_bound(cu_tuples_first.begin(), cu_tuples_first.end(), search_begin,
                            search_begin + (int)powf(this->m_gridSize, this->m_dim), m_cu_data_map.bucket_end.begin());

        addDataMapFromDevice();

        skip += num_of_samples_compute;
    }
}

template<typename T>
using cu_complex = typename cuComplexVector<T>::value_type;

__constant__ float d_kernel[512];

template<typename T> __global__
void gridding_kernel(const cu_complex<T> *kData, const T *dcf, cu_complex<T> *out,
                     const unsigned *bucket_begin, const unsigned *bucket_end, const SampleTuple *tuples_last,
                     float half_W, size_t num_of_data_compute)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= num_of_data_compute)
        return;

    constexpr int kernel_length = sizeof(d_kernel) / sizeof(d_kernel[0]);

    cu_complex<T> data {0, 0};
    for (unsigned i = bucket_begin[index]; i < bucket_end[index]; ++i)
    {
        auto delta = tuples_last[i].delta;
        auto sample_idx = tuples_last[i].sample_idx;

        if (delta < half_W)
        {
            int ki = (int)(delta / half_W * (kernel_length - 1));
            data.x += kData[sample_idx].x * dcf[sample_idx] * d_kernel[ki];
            data.y += kData[sample_idx].y * dcf[sample_idx] * d_kernel[ki];
        }
    }

    out[index] = data;
}

template<typename T>
cuComplexVector<T> *cuGridLut<T>::griddingChannel(cuReconData<T> &reconData, int channel)
{
    const cuComplexVector<T> *kData = reconData.cuGetChannelData(channel);
    const cuVector<T> *dcf = reconData.cuGetDcf();

    auto out = new cuComplexVector<T>((int)powf(this->m_gridSize, this->m_dim));

    auto d_kData = thrust::raw_pointer_cast(kData->data());
    auto d_out = thrust::raw_pointer_cast(out->data());
    auto d_dcf = thrust::raw_pointer_cast(dcf->data());

    auto d_bucket_begin = thrust::raw_pointer_cast(m_cu_bucket_begin->data());
    auto d_bucket_end = thrust::raw_pointer_cast(m_cu_bucket_end->data());

    auto kernel = this->m_kernel.getKernelData();
    assert(kernel->size() == sizeof(d_kernel) / sizeof(d_kernel[0]));
    cudaMemcpyToSymbol(d_kernel, kernel->data(), kernel->size() * sizeof(float));

    size_t chunk_size = 8000;
    size_t blockSize = 256;
    size_t gridSize = (size_t)ceil((float)chunk_size / blockSize);

    auto tuples_it = m_tuples_last->cbegin();
    auto bucket_begin_it = m_cu_bucket_begin->cbegin();
    auto bucket_end_it = m_cu_bucket_end->cbegin();
    size_t skip = 0;

    while (skip < out->size())
    {
        size_t num_of_data_compute = std::min(chunk_size, out->size() - skip);

        auto tuples_it_first = tuples_it + *(bucket_begin_it + skip);
        auto tuples_it_last = tuples_it + *(bucket_end_it + skip + num_of_data_compute - 1);

        if (tuples_it_first < tuples_it_last)
        {
            thrust::device_vector<SampleTuple> tuples_last(tuples_it_first, tuples_it_last);
            auto d_tuples_last = thrust::raw_pointer_cast(tuples_last.data()) - *(bucket_begin_it + skip);

            gridding_kernel<T><<<gridSize, blockSize>>>(d_kData, d_dcf, d_out + skip, d_bucket_begin + skip, d_bucket_end + skip, d_tuples_last,
                                                        this->m_kernel.getKernelWidth() / 2.0, num_of_data_compute);
            cudaDeviceSynchronize();
        }
        skip += num_of_data_compute;
    }
    return out;
}

template class cuGridLut<float>;
