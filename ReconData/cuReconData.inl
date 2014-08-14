
template<typename T>
cuReconData<T>::cuReconData(int size)
    : ReconData<thrust::device_vector, T>(size)
{
}

template<typename T>
void cuReconData<T>::addTrajIndexBlock(cuVector<T> &index)
{

}
