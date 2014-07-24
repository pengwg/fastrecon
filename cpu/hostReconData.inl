
template<typename T>
hostReconData<T>::hostReconData(int size)
    : ReconData<std::vector, T>(size)
{
}

template<typename T>
void hostReconData<T>::transformLocalTrajComp(float translation, float scale, int comp)
{
    for (auto &sample : *this->m_traj[comp])
    {
        sample = (sample + translation) * scale;
    }
}
