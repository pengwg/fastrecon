
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
void cuReconData<T>::cuPreprocess(T half_W)
{
    auto cells_per_sample = new thrust::device_vector<int>;
    this->cuComputeCellsPerSample(*this->getTraj(), half_W, *cells_per_sample);
    delete cells_per_sample;
}
