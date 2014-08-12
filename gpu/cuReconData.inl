
template<typename T>
cuReconData<T>::cuReconData(int size)
    : ReconData<thrust::device_vector, T>(size)
{
}

template<typename T>
void cuReconData<T>::transformLocalTraj(float translation, float scale)
{
    this->cuScale(*this->m_traj, translation, scale);
}

template<typename T>
void cuReconData<T>::addTrajIndexBlock(cuReconData::cuVector &index)
{

}

template<typename T>
void cuReconData<T>::preprocess(int reconSize, T half_W, thrust::host_vector<int> *tuples_last,
                                thrust::host_vector<unsigned> *bucket_begin, thrust::host_vector<unsigned> *bucket_end) const
{
    this->cuPreprocess(*this->getTraj(), reconSize, half_W, tuples_last, bucket_begin, bucket_end);
}
