
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

template<typename T>
void cuReconData<T>::cuComputeCellsPerSample(T half_W, thrust::device_vector<unsigned> &cells_per_sample) const
{
    assert(this->rcDim() > 0);
    this->cuComputeSampleCoverage(*this->getTrajComponent(0), half_W, cells_per_sample);

    thrust::device_vector<unsigned> temp;
    for (int i = 1; i < this->rcDim(); ++i)
    {
        this->cuComputeSampleCoverage(*this->getTrajComponent(i), half_W, temp);
        this->cuMultiplies(temp, cells_per_sample, cells_per_sample);
    }
}

template<typename T>
void cuReconData<T>::addTrajIndexBlock(cuReconData::cuVector &index)
{

}

template<typename T>
void cuReconData<T>::cuPreprocess(T half_W)
{
    auto cells_per_sample = new thrust::device_vector<unsigned>;
    cuComputeCellsPerSample(half_W, *cells_per_sample);
    delete cells_per_sample;
}
