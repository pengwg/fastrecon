#include <iostream>

#include "ReconData.h"

template <typename T>
void ReconData<T>::setTraj(T *traj)
{
    if (channels() > 0 && m_size != traj->size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_size = traj->size();
    m_rcDim = sizeof(traj->at(0).pos) / sizeof(traj->at(0).pos[0]);

    m_traj.reset(traj);
}

template <typename T>
void ReconData<T>::addChannelData(KData *data)
{
    if (m_size != 0 && m_size != data->size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_size = data->size();
    m_kDataMultiChannel.push_back(std::shared_ptr<KData>(data));
}

template class ReconData<Traj2D>;
template class ReconData<Traj3D>;

