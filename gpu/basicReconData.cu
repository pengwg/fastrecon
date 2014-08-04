#include <thrust/transform.h>
#include "basicReconData.h"

template<typename T>
void basicReconData<T>::thrust_scale(thrust::device_vector<T> &traj, T translation, T scale) const
{
    thrust::transform(traj.begin(), traj.end(), traj.begin(), scale_functor(translation, scale));
}

template<typename T>
void basicReconData<T>::cuComputeSampleCoverage(const thrust::device_vector<T> &traj, T half_W, thrust::device_vector<unsigned> &cell_coverage) const
{
    cell_coverage.resize(traj.size());
    thrust::transform(traj.begin(), traj.end(), cell_coverage.begin(), compute_sample_coverage(half_W));
}

template<typename T>
void basicReconData<T>::cuMultiplies(const thrust::device_vector<unsigned> &in1, const thrust::device_vector<unsigned> &in2, thrust::device_vector<unsigned> &out) const
{
    thrust::transform(in1.begin(), in1.end(), in2.begin(), out.begin(), thrust::multiplies<unsigned>());
}

template void basicReconData<float>::thrust_scale(thrust::device_vector<float>&, float, float) const;
template void basicReconData<float>::cuComputeSampleCoverage(const thrust::device_vector<float>&, float, thrust::device_vector<unsigned> &) const;
template void basicReconData<float>::cuMultiplies(const thrust::device_vector<unsigned>&, const thrust::device_vector<unsigned>&, thrust::device_vector<unsigned>&) const;
