//#include "cudaFunctions.h"

template<typename T>
cuReconData<T>::cuReconData(int size)
    : ReconData<thrust::device_vector, T>(size)
{
}

template<typename T>
void cuReconData<T>::transformLocalTrajComp(float translation, float scale, int comp)
{
    this->thrust_scale(*this->m_traj[comp], translation, scale);
}
