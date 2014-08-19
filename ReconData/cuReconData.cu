#include <thrust/transform.h>

#include "cuReconData.h"

template<typename T>
cuReconData<T>::cuReconData(int size)
    : ReconData<T>(size)
{

}

template<typename T>
const cuComplexVector<T> *cuReconData<T>::cuGetChannelData(int channel)
{
    if (channel == m_channel_in_device)
    {
        return m_cu_kData.get();
    }

    if (channel < this->channels())
    {
        typedef hostVector<typename cuComplexVector<T>::value_type> interim_type;
        auto ptr = reinterpret_cast<const interim_type *>(this->getChannelData(channel));
        auto cu_kData = new cuComplexVector<T>(*ptr);
        m_cu_kData.reset(cu_kData);
        m_channel_in_device = channel;
        return cu_kData;
    }

    return nullptr;
}

template<typename T>
cuReconData<T>::cuTrajVector *cuReconData<T>::cuGetTraj()
{
    if (m_cu_traj != nullptr)
    {
        return m_cu_traj.get();
    }

    auto cu_traj = new cuTrajVector(*this->m_traj.get());
    m_cu_traj.reset(cu_traj);
    return cu_traj;
}

template<typename T>
const cuVector<T> *cuReconData<T>::cuGetDcf()
{
    if (m_cu_dcf != nullptr)
    {
        return m_cu_dcf.get();
    }

    auto cu_dcf = new cuVector<T>(*this->m_dcf.get());
    m_cu_dcf.reset(cu_dcf);
    return cu_dcf;
}

template<typename T>
void cuReconData<T>::clear()
{
    ReconData<T>::clear();

    m_cu_dcf.reset(nullptr);
    m_cu_kData.reset(nullptr);
    m_cu_traj.reset(nullptr);
    m_channel_in_device = -1;
}

template<typename T>
void cuReconData<T>::transformLocalTraj(T translation, T scale)
{
    cuTrajVector *traj = cuGetTraj();
    thrust::transform(traj->begin(), traj->end(), traj->begin(), scale_functor<T>(translation, scale, this->rcDim()));

    thrust::copy(traj->begin(), traj->end(), this->m_traj->begin());

    //thrust::host_vector<Point<T>> host_traj(*traj);
    //std::copy(host_traj.begin(), host_traj.end(), this->m_traj->begin());
    //std::memcpy(this->m_traj->data(), host_traj.data(), host_traj.size() * sizeof(Point<T>));
}

template class cuReconData<float>;
