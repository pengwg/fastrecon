#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/system/omp/execution_policy.h>

#include "basicReconData.h"

template<typename T>
struct scale_functor
{
    const T _a, _b;
    int _dim;
    scale_functor(T a, T b, int dim) : _a(a), _b(b), _dim(dim) {}
    __host__ __device__
    Point<T> operator() (const Point<T> &p) const {
        Point<T> p0;
        for( int d = 0; d < _dim; ++d) {
            p0.x[d] = (p.x[d] + _a) * _b;
        }
        return p0;
    }
};

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
void write_pairs_kernel(const Point<T> *traj, unsigned *tuple_index, int *tuples_first, int *tuples_last, int reconSize, T half_W, int num_samples)
{
    unsigned sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
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

void savePreprocess(const thrust::host_vector<unsigned> *first, const thrust::host_vector<unsigned> *last)
{
    QFile file("bucket_begin.dat");
    file.open(QIODevice::WriteOnly);
    file.write((char *)thrust::raw_pointer_cast(first->data()), first->size() * sizeof(unsigned));
    file.close();

    file.setFileName("bucket_end.dat");
    file.open(QIODevice::WriteOnly);
    file.write((char *)thrust::raw_pointer_cast(last->data()), last->size() * sizeof(unsigned));
    file.close();
}

bool loadPreprocess(thrust::host_vector<int> *first, thrust::host_vector<int> *last, unsigned length)
{
    first->resize(length);
    last->resize(length);

    QFile file("tuples_first.dat");
    file.open(QIODevice::ReadOnly);
    auto count = file.read((char *)thrust::raw_pointer_cast(first->data()), length * sizeof(int));
    file.close();

    if (count != (int)(length * sizeof(int)))
    {
        std::cout << "Error: wrong data size in tuples_first.dat" << std::endl;
        return false;
    }

    file.setFileName("tuples_last.dat");
    file.open(QIODevice::ReadOnly);
    count = file.read((char *)thrust::raw_pointer_cast(last->data()), length * sizeof(int));
    file.close();

    if (count != (int)(length * sizeof(int)))
    {
        std::cout << "Error: wrong data size in tuples_last.dat" << std::endl;
        return false;
    }

    return true;
}

template<typename T>
void basicReconData<T>::cuScale(thrust::device_vector<Point<T> > &traj, T translation, T scale) const
{
    thrust::transform(traj.begin(), traj.end(), traj.begin(), scale_functor<T>(translation, scale, m_dim));
}

template<typename T>
void basicReconData<T>::cuPreprocess(const thrust::device_vector<Point<T> > &traj, int reconSize, T half_W) const
{
    auto cells_per_sample = new thrust::device_vector<unsigned> (traj.size());
    thrust::transform(traj.begin(), traj.end(), cells_per_sample->begin(), compute_num_cells_per_sample<T>(half_W, m_dim));

    auto tuple_index = new thrust::device_vector<unsigned> (traj.size());
    thrust::inclusive_scan(cells_per_sample->begin(), cells_per_sample->end(), tuple_index->begin(), thrust::plus<unsigned> ());
    delete cells_per_sample;

    tuple_index->insert(tuple_index->begin(), 0);
    unsigned num_of_pairs_total = tuple_index->back();
    std::cout << " Traj size: " << traj.size() << "; Number of pairs: " << num_of_pairs_total << std::endl;

    auto tuples_first_h = new thrust::host_vector<int>;
    auto tuples_last_h = new thrust::host_vector<int>;

    if (loadPreprocess(tuples_first_h, tuples_last_h, num_of_pairs_total))
    {
        std::cout << "Loaded preprocessed data from disk." << std::endl;

        auto bucket_begin = new thrust::host_vector<unsigned>(powf(reconSize, m_dim));
        auto bucket_end   = new thrust::host_vector<unsigned>(powf(reconSize, m_dim));

        thrust::counting_iterator<unsigned int> search_begin(0);
        thrust::lower_bound(tuples_first_h->begin(), tuples_first_h->end(), search_begin,
                    search_begin + (int)powf(reconSize, m_dim), bucket_begin->begin());
        thrust::upper_bound(tuples_first_h->begin(), tuples_first_h->end(), search_begin,
                    search_begin + (int)powf(reconSize, m_dim), bucket_end->begin());

        savePreprocess(bucket_begin, bucket_end);

        delete bucket_begin;
        delete bucket_end;

        delete tuple_index;
        delete tuples_first_h;
        delete tuples_last_h;
        return;
    }

    std::cout << "Preprocesse data for CUDA..." << std::endl;

    auto tuples_first = new thrust::device_vector<int>;
    auto tuples_last = new thrust::device_vector<int>;

    unsigned chunk_size = 100000;

    unsigned skip = 0;
    while (skip < traj.size())
    {
        unsigned num_of_samples_compute = min(chunk_size, unsigned(traj.size()) - skip);
        unsigned num_of_pairs = (*tuple_index)[skip + num_of_samples_compute] - (*tuple_index)[skip];
        std::cout << "num_of_pairs compute: " << num_of_pairs << std::endl;
        tuples_first->resize(num_of_pairs);
        tuples_last->resize(num_of_pairs);

        const Point<T> *traj_ptr = thrust::raw_pointer_cast(traj.data()) + skip;
        unsigned *tuple_index_ptr = thrust::raw_pointer_cast(tuple_index->data()) + skip;
        int *tuples_first_ptr = thrust::raw_pointer_cast(tuples_first->data()) - (*tuple_index)[skip];
        int *tuples_last_ptr = thrust::raw_pointer_cast(tuples_last->data()) - (*tuple_index)[skip];

        int blockSize = 256;
        int gridSize = (int)ceil((double)chunk_size / blockSize);
        write_pairs_kernel<T><<<gridSize, blockSize>>>(traj_ptr, tuple_index_ptr, tuples_first_ptr, tuples_last_ptr,
                                                       reconSize, half_W, num_of_samples_compute);

        thrust::sort_by_key(tuples_first->begin(), tuples_first->end(), tuples_last->begin());

        tuples_first_h->insert(tuples_first_h->end(), tuples_first->begin(), tuples_first->end());
        tuples_last_h->insert(tuples_last_h->end(), tuples_last->begin(), tuples_last->end());

        skip += num_of_samples_compute;
    }

    delete tuples_first;
    delete tuples_last;

    thrust::sort_by_key(thrust::system::omp::par, tuples_first_h->begin(), tuples_first_h->end(), tuples_last_h->begin());

    auto bucket_begin = new thrust::host_vector<unsigned>(powf(reconSize, m_dim));
    auto bucket_end   = new thrust::host_vector<unsigned>(powf(reconSize, m_dim));

    thrust::counting_iterator<unsigned int> search_begin(0);
    thrust::lower_bound(tuples_first_h->begin(), tuples_first_h->end(), search_begin,
                search_begin + (int)powf(reconSize, m_dim), bucket_begin->begin());
    thrust::upper_bound(tuples_first_h->begin(), tuples_first_h->end(), search_begin,
                search_begin + (int)powf(reconSize, m_dim), bucket_end->begin());

    savePreprocess(bucket_begin, bucket_end);

    delete bucket_begin;
    delete bucket_end;

    delete tuple_index;
    delete tuples_first_h;
    delete tuples_last_h;
}

template void basicReconData<float>::cuScale(thrust::device_vector<Point<float> >&, float, float) const;
template void basicReconData<float>::cuPreprocess(const thrust::device_vector<Point<float> >&, int, float) const;
