#include <thrust/transform.h>
#include "basicReconData.h"

template<typename T>
void basicReconData<T>::thrust_scale(thrust::device_vector<T> &traj, T translation, T scale)
{
    thrust::transform(traj.begin(), traj.end(), traj.begin(), scale_functor(translation, scale));
}

template void basicReconData<float>::thrust_scale(thrust::device_vector<float>&, float, float);

