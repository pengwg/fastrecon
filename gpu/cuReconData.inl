
template<typename T>
cuReconData<T>::cuReconData(int size)
    : ReconData<thrust::device_vector, T>(size)
{
}

template<typename T>
void cuReconData<T>::transformLocalTraj(float translation, float scale)
{
    this->thrust_scale(*this->m_traj, translation, scale);
}

template<typename T>
void cuReconData<T>::addTrajIndexBlock(cuReconData::cuVector &index)
{

}

template<typename T>
void cuReconData<T>::preprocess(T half_W)
{
    this->cuPreprocess(*this->getTraj(), half_W);
}
