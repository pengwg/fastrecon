#include <thrust/transform.h>

#include "cuReconData.h"

template<typename T>
cuReconData<T>::cuReconData(int size)
    : ReconData<T>(size)
{

}

template<typename T>
void cuReconData<T>::transformLocalTraj(T translation, T scale)
{
    auto traj = m_cu_traj.get();
    thrust::transform(traj->begin(), traj->end(), traj->begin(), scale_functor<T>(translation, scale, this->rcDim()));
}

template class cuReconData<float>;
