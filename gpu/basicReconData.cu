#include <thrust/transform.h>
#include "basicReconData.h"

template<typename T>
void basicReconData<T>::thrust_scale(thrust::device_vector<Point<T> > &traj, T translation, T scale) const
{
    thrust::transform(traj.begin(), traj.end(), traj.begin(), scale_functor(translation, scale, m_dim));
}

template<typename T>
void basicReconData<T>::cuPreprocess(const thrust::device_vector<Point<T> > &traj, T half_W) const
{
    auto cells_per_sample = new thrust::device_vector<int> (traj.size());
    thrust::transform(traj.begin(), traj.end(), cells_per_sample->begin(), compute_num_cells_per_sample(half_W, m_dim));
    delete cells_per_sample;
}

template void basicReconData<float>::thrust_scale(thrust::device_vector<Point<float> >&, float, float) const;
template void basicReconData<float>::cuPreprocess(const thrust::device_vector<Point<float> >&, float) const;
