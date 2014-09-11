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

void savePreprocess(const thrust::host_vector<int> *tuples_first, const thrust::host_vector<SampleTuple> *tuples_last, const thrust::host_vector<unsigned> *bucket_begin, const thrust::host_vector<unsigned> *bucket_end)
{
    QFile file("tuples_last.dat");
    file.open(QIODevice::WriteOnly);
    file.write((char *)thrust::raw_pointer_cast(tuples_last->data()), tuples_last->size() * sizeof(SampleTuple));
    file.close();

    file.setFileName("tuples_first.dat");
    file.open(QIODevice::WriteOnly);
    file.write((char *)thrust::raw_pointer_cast(tuples_first->data()), tuples_first->size() * sizeof(int));
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

bool loadPreprocess(thrust::host_vector<int> *tuples_first, thrust::host_vector<SampleTuple> *tuples_last, thrust::host_vector<unsigned> *bucket_begin, thrust::host_vector<unsigned> *bucket_end)
{
    QFile file("tuples_last.dat");
    if (!file.exists())
        return false;
    auto length = file.size() / sizeof(SampleTuple);
    tuples_last->resize(length);

    file.open(QIODevice::ReadOnly);
    file.read((char *)thrust::raw_pointer_cast(tuples_last->data()), file.size());
    file.close();

    file.setFileName("tuples_first.dat");
    if (!file.exists())
        return false;
    length = file.size() / sizeof(int);
    tuples_first->resize(length);

    file.open(QIODevice::ReadOnly);
    file.read((char *)thrust::raw_pointer_cast(tuples_first->data()), file.size());
    file.close();

    file.setFileName("bucket_begin.dat");
    if (!file.exists())
        return false;
    length = file.size() / sizeof(unsigned);
    bucket_begin->resize(length);

    file.open(QIODevice::ReadOnly);
    file.read((char *)thrust::raw_pointer_cast(bucket_begin->data()), file.size());
    file.close();

    file.setFileName("bucket_end.dat");
    if (!file.exists())
        return false;
    length = file.size() / sizeof(unsigned);
    bucket_end->resize(length);

    file.open(QIODevice::ReadOnly);
    file.read((char *)thrust::raw_pointer_cast(bucket_end->data()), file.size());
    file.close();

    return true;
}

template<typename T>
void cuGridLut<T>::plan(const cuVector<Point<T>> &traj)
{
    auto tuples_first_h = new thrust::host_vector<int>;
    auto tuples_last_h = new thrust::host_vector<SampleTuple>;
    auto bucket_begin_h = new thrust::host_vector<unsigned>;
    auto bucket_end_h   = new thrust::host_vector<unsigned>;

    T half_W = this->m_kernel.getKernelWidth() / 2.0;

    auto cells_per_sample = new thrust::device_vector<unsigned> (traj.size());
    thrust::transform(traj.begin(), traj.end(), cells_per_sample->begin(), compute_num_cells_per_sample<T>(half_W, this->m_dim));

    auto tuple_index = new thrust::device_vector<unsigned> (traj.size());
    thrust::inclusive_scan(cells_per_sample->begin(), cells_per_sample->end(), tuple_index->begin(), thrust::plus<unsigned> ());
    delete cells_per_sample;

    tuple_index->insert(tuple_index->begin(), 0);
    unsigned num_of_pairs_total = tuple_index->back();

    bool loadSuccess = false;
    if (loadPreprocess(tuples_first_h, tuples_last_h, bucket_begin_h, bucket_end_h))
    {
        std::cout << "Loaded preprocessed data from disk." << std::endl;

        if (tuples_last_h->size() != num_of_pairs_total || bucket_begin_h->size() != powf(this->m_gridSize, this->m_dim) ||
                bucket_end_h->size() != powf(this->m_gridSize, this->m_dim))
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

        tuples_first_h->clear();
        tuples_last_h->clear();

        auto tuples_first = new thrust::device_vector<int>;
        auto tuples_last = new thrust::device_vector<SampleTuple>;

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
            SampleTuple *tuples_last_ptr = thrust::raw_pointer_cast(tuples_last->data()) - (*tuple_index)[skip];

            write_pairs_kernel<T><<<gridSize, blockSize>>>(traj_ptr, tuple_index_ptr, tuples_first_ptr, tuples_last_ptr,
                                                           this->m_gridSize, half_W, num_of_samples_compute, skip);

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

        bucket_begin_h->resize(powf(this->m_gridSize, this->m_dim));
        bucket_end_h->resize(powf(this->m_gridSize, this->m_dim));

        std::cout << "Generate buckets... " << std::flush;
        thrust::counting_iterator<int> search_begin(0);
        thrust::lower_bound(thrust::system::omp::par, tuples_first_h->begin(), tuples_first_h->end(), search_begin,
                            search_begin + (int)powf(this->m_gridSize, this->m_dim), bucket_begin_h->begin());
        thrust::upper_bound(thrust::system::omp::par, tuples_first_h->begin(), tuples_first_h->end(), search_begin,
                            search_begin + (int)powf(this->m_gridSize, this->m_dim), bucket_end_h->begin());

        std::cout << "Save data to disk... " << std::flush;
        savePreprocess(tuples_first_h, tuples_last_h, bucket_begin_h, bucket_end_h);
        std::cout << "done." << std::endl;
    }

    std::cout << " Traj size: " << traj.size() << ", Image size: " << bucket_begin_h->size() << ", Number of pairs: " << num_of_pairs_total << std::endl;

    /*auto it = bucket_end_h->cbegin();
    for (int i = 0; i < 2024; i++)
        std::cout << *(it++) << ' ';
    std::cout << std::endl;*/

    auto bucket_begin = new thrust::device_vector<unsigned>(*bucket_begin_h);
    auto bucket_end   = new thrust::device_vector<unsigned>(*bucket_end_h);

    delete tuples_first_h;
    delete bucket_begin_h;
    delete bucket_end_h;

    m_tuples_last.reset(tuples_last_h);
    m_cu_bucket_begin.reset(bucket_begin);
    m_cu_bucket_end.reset(bucket_end);
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
