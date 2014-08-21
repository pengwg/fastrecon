#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/system/omp/execution_policy.h>

#include "GridLut.h"

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
void write_pairs_kernel(const Point<T> *traj, unsigned *tuple_index, int *tuples_first, unsigned *tuples_last, int reconSize, T half_W, size_t num_samples)
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
        //printf("sample: %d, offset: %d, lb[0]: %d, ub[0]: %d\n", sample_idx, write_offset, lb[2], ub[2]);

    for (int z = lb[2]; z <= ub[2]; ++z)
    {
        for (int y = lb[1]; y <= ub[1]; ++y)
        {
            for (int x = lb[0]; x <= ub[0]; ++x)
            {
                int matrix_index = x + y * reconSize + z * reconSize * reconSize;
                tuples_first[write_offset + counter] = matrix_index;
                tuples_last[write_offset + counter] = sample_idx;
                ++counter;
            }
        }
    }

}

void savePreprocess(const thrust::host_vector<unsigned> *tuples_last, const thrust::host_vector<unsigned> *bucket_begin, const thrust::host_vector<unsigned> *bucket_end)
{
    QFile file("tuples_last.dat");
    file.open(QIODevice::WriteOnly);
    file.write((char *)thrust::raw_pointer_cast(tuples_last->data()), tuples_last->size() * sizeof(unsigned));
    file.close();

    file.setFileName("bucket_begin.dat");
    file.open(QIODevice::WriteOnly);
    file.write((char *)thrust::raw_pointer_cast(bucket_begin->data()), bucket_begin->size() * sizeof(unsigned));
    file.close();

    file.setFileName("bucket_end.dat");
    file.open(QIODevice::WriteOnly);
    file.write((char *)thrust::raw_pointer_cast(bucket_end->data()), bucket_end->size() * sizeof(unsigned));
    file.close();
}

bool loadPreprocess(thrust::host_vector<unsigned> *tuples_last, thrust::host_vector<unsigned> *bucket_begin, thrust::host_vector<unsigned> *bucket_end)
{
    QFile file("tuples_last.dat");
    if (!file.exists())
        return false;
    auto length = file.size() / sizeof(unsigned);
    tuples_last->resize(length);

    file.open(QIODevice::ReadOnly);
    file.read((char *)thrust::raw_pointer_cast(tuples_last->data()), length * sizeof(unsigned));
    file.close();

    file.setFileName("bucket_begin.dat");
    if (!file.exists())
        return false;
    length = file.size() / sizeof(unsigned);
    bucket_begin->resize(length);

    file.open(QIODevice::ReadOnly);
    file.read((char *)thrust::raw_pointer_cast(bucket_begin->data()), length * sizeof(unsigned));
    file.close();

    file.setFileName("bucket_end.dat");
    if (!file.exists())
        return false;
    length = file.size() / sizeof(unsigned);
    bucket_end->resize(length);

    file.open(QIODevice::ReadOnly);
    file.read((char *)thrust::raw_pointer_cast(bucket_end->data()), length * sizeof(unsigned));
    file.close();

    return true;
}

template<typename T>
void GridLut<T>::cuPlan(const cuVector<Point<T>> &traj)
{
    auto tuples_last_h = new thrust::host_vector<unsigned>;
    auto bucket_begin_h = new thrust::host_vector<unsigned>;
    auto bucket_end_h   = new thrust::host_vector<unsigned>;

    T half_W = m_kernel.getKernelWidth() / 2.0;

    auto cells_per_sample = new thrust::device_vector<unsigned> (traj.size());
    thrust::transform(traj.begin(), traj.end(), cells_per_sample->begin(), compute_num_cells_per_sample<T>(half_W, m_dim));

    auto tuple_index = new thrust::device_vector<unsigned> (traj.size());
    thrust::inclusive_scan(cells_per_sample->begin(), cells_per_sample->end(), tuple_index->begin(), thrust::plus<unsigned> ());
    delete cells_per_sample;

    tuple_index->insert(tuple_index->begin(), 0);
    unsigned num_of_pairs_total = tuple_index->back();
    std::cout << " Traj size: " << traj.size() << "; Number of pairs: " << num_of_pairs_total << std::endl;

    bool loadSuccess = false;
    if (loadPreprocess(tuples_last_h, bucket_begin_h, bucket_end_h))
    {
        std::cout << "Loaded preprocessed data from disk." << std::endl;

        if (tuples_last_h->size() != num_of_pairs_total || bucket_begin_h->size() != powf(m_gridSize, m_dim) ||
                bucket_end_h->size() != powf(m_gridSize, m_dim))
        {
            std::cout << "Wrong size, recompute data... " << std::endl;
        }
        else
        {
            loadSuccess = true;
        }
    }

    if (!loadSuccess)
    {
        std::cout << "Preprocesse data for CUDA..." << std::endl;

        auto tuples_first_h = new thrust::host_vector<int>;
        tuples_last_h->clear();

        auto tuples_first = new thrust::device_vector<int>;
        auto tuples_last = new thrust::device_vector<unsigned>;

        size_t chunk_size = 100000;
        size_t blockSize = 256;
        size_t gridSize = (size_t)ceil((float)chunk_size / blockSize);

        size_t skip = 0;
        while (skip < traj.size())
        {
            size_t num_of_samples_compute = std::min(chunk_size, traj.size() - skip);
            unsigned num_of_pairs = (*tuple_index)[skip + num_of_samples_compute] - (*tuple_index)[skip];
            std::cout << "num_of_pairs compute: " << num_of_pairs << std::endl;
            tuples_first->resize(num_of_pairs);
            tuples_last->resize(num_of_pairs);

            const Point<T> *traj_ptr = thrust::raw_pointer_cast(traj.data()) + skip;
            unsigned *tuple_index_ptr = thrust::raw_pointer_cast(tuple_index->data()) + skip;
            int *tuples_first_ptr = thrust::raw_pointer_cast(tuples_first->data()) - (*tuple_index)[skip];
            unsigned *tuples_last_ptr = thrust::raw_pointer_cast(tuples_last->data()) - (*tuple_index)[skip];

            write_pairs_kernel<T><<<gridSize, blockSize>>>(traj_ptr, tuple_index_ptr, tuples_first_ptr, tuples_last_ptr,
                                                           m_gridSize, half_W, num_of_samples_compute);

            thrust::sort_by_key(tuples_first->begin(), tuples_first->end(), tuples_last->begin());

            tuples_first_h->insert(tuples_first_h->end(), tuples_first->begin(), tuples_first->end());
            tuples_last_h->insert(tuples_last_h->end(), tuples_last->begin(), tuples_last->end());

            skip += num_of_samples_compute;
        }

        delete tuple_index;
        delete tuples_first;
        delete tuples_last;

        std::cout << "Sort tuples... " << std::flush;
        thrust::sort_by_key(thrust::system::omp::par, tuples_first_h->begin(), tuples_first_h->end(), tuples_last_h->begin());

        bucket_begin_h->resize(powf(m_gridSize, m_dim));
        bucket_end_h->resize(powf(m_gridSize, m_dim));

        std::cout << "Generate buckets... " << std::flush;
        thrust::counting_iterator<unsigned> search_begin(0);
        thrust::lower_bound(thrust::system::omp::par, tuples_first_h->begin(), tuples_first_h->end(), search_begin,
                            search_begin + (int)powf(m_gridSize, m_dim), bucket_begin_h->begin());
        thrust::upper_bound(thrust::system::omp::par, tuples_first_h->begin(), tuples_first_h->end(), search_begin,
                            search_begin + (int)powf(m_gridSize, m_dim), bucket_end_h->begin());

        std::cout << "Save data to disk... " << std::flush;
        savePreprocess(tuples_last_h, bucket_begin_h, bucket_end_h);
        std::cout << "done." << std::endl;

        delete tuples_first_h;
    }

    auto bucket_begin = new thrust::device_vector<unsigned>(*bucket_begin_h);
    auto bucket_end   = new thrust::device_vector<unsigned>(*bucket_end_h);

    delete bucket_begin_h;
    delete bucket_end_h;

    m_tuples_last.reset(tuples_last_h);
    m_cu_bucket_begin.reset(bucket_begin);
    m_cu_bucket_end.reset(bucket_end);
}

template<typename T>
using cu_complex = typename cuComplexVector<T>::value_type;

template<typename T> __global__
void gridding_kernel(const cu_complex<T> *kData, cu_complex<T> *out)
{

}

template<typename T>
cuComplexVector<T> *GridLut<T>::griddingChannel(cuReconData<T> &reconData, int channel)
{
    const cuComplexVector<T> *kData = reconData.cuGetChannelData(channel);
    auto out = new cuComplexVector<T>(powf(m_gridSize, m_dim));

    auto d_kData = thrust::raw_pointer_cast(kData->data());
    auto d_out = thrust::raw_pointer_cast(out->data());

    auto tuples_last_ptr = thrust::raw_pointer_cast(m_tuples_last.get()->data());
    auto bucket_begin_ptr = thrust::raw_pointer_cast(m_cu_bucket_begin.get()->data());
    auto bucket_end_ptr = thrust::raw_pointer_cast(m_cu_bucket_end.get()->data());

    gridding_kernel<T><<<gridSize, blockSize>>>(d_kData, d_out);

    return out;
}

template void GridLut<float>::cuPlan(const thrust::device_vector<Point<float>> &traj);
template cuComplexVector<float> *GridLut<float>::griddingChannel(cuReconData<float> &reconData, int channel);
