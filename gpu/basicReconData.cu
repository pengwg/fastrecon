#include <thrust/transform.h>
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

    unsigned write_offset = sample_idx == 0 ? 0 : tuple_index[sample_idx - 1];
    int counter = 0;

    //if (sample_idx < 10)
        //printf("sample: %d, offset: %d, lb[0]: %d, ub[0]: %d\n", sample_idx, write_offset, lb[2], ub[2]);

    for (int z = lb[2]; z <= ub[2]; ++z)
    {
        for (int y = lb[1]; y <= ub[1]; ++y)
        {
            for (int x = lb[0]; x <= ub[0]; ++x)
            {
                tuples_first[write_offset + counter] = x + y * reconSize + z * reconSize * reconSize;
                tuples_last[write_offset + counter] = sample_idx;
                ++counter;
            }
        }
    }

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

    std::cout << " Traj size: " << traj.size() << " Number of pairs: " << tuple_index->back() << std::endl;

    unsigned chunk_size = 524288;
    unsigned num_of_pairs = tuple_index->data()[chunk_size];
    auto tuples_first = new thrust::device_vector<int> (num_of_pairs);
    auto tuples_last = new thrust::device_vector<int> (num_of_pairs);

    const Point<T> *traj_ptr = thrust::raw_pointer_cast(traj.data());
    unsigned *tuple_index_ptr = thrust::raw_pointer_cast(tuple_index->data());
    int *tuples_first_ptr = thrust::raw_pointer_cast(tuples_first->data());
    int *tuples_last_ptr = thrust::raw_pointer_cast(tuples_last->data());

    int blockSize = 256;
    int gridSize = (int)ceil((double)chunk_size / blockSize);
    write_pairs_kernel<T><<<gridSize, blockSize>>>(traj_ptr, tuple_index_ptr, tuples_first_ptr, tuples_last_ptr, reconSize, half_W, chunk_size);

    delete tuples_first;
    delete tuples_last;
    delete tuple_index;
}

template void basicReconData<float>::cuScale(thrust::device_vector<Point<float> >&, float, float) const;
template void basicReconData<float>::cuPreprocess(const thrust::device_vector<Point<float> >&, int, float) const;
