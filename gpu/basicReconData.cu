#include <thrust/transform.h>
#include "basicReconData.h"

template<typename T>
void basicReconData<T>::thrust_scale(thrust::device_vector<Point<T> > &traj, T translation, T scale) const
{
    thrust::transform(traj.begin(), traj.end(), traj.begin(), scale_functor(translation, scale, m_dim));
}

template<typename T>
void basicReconData<T>::cuComputeCellsPerSample(const thrust::device_vector<Point<T> > &traj, T half_W, thrust::device_vector<int> &cell_coverage) const
{
    cell_coverage.resize(traj.size());
    thrust::transform(traj.begin(), traj.end(), cell_coverage.begin(), compute_num_cells_per_sample(half_W, m_dim));
}

template void basicReconData<float>::thrust_scale(thrust::device_vector<Point<float> >&, float, float) const;
template void basicReconData<float>::cuComputeCellsPerSample(const thrust::device_vector<Point<float> >&, float, thrust::device_vector<int> &) const;
