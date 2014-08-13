
template<typename T>
hostReconData<T>::hostReconData(int size)
    : ReconData<std::vector, T>(size)
{
}

template<typename T>
void hostReconData<T>::transformLocalTraj(T translation, T scale)
{
    for (auto &sample : *this->m_traj)
    {
        for (int comp = 0; comp < this->m_dim; ++comp)
            sample.x[comp] = (sample.x[comp] + translation) * scale;
    }
}
