#include <iostream>

#include "ReconData.h"

template <int N>
void ReconData<N>::setTraj(Traj<N> *traj)
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

template <int N>
void ReconData<N>::addChannelData(ComplexVector *data)
{
    if (m_size != 0 && m_size != data->size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_size = data->size();
    m_kDataMultiChannel.push_back(std::shared_ptr<ComplexVector>(data));
}

template <int N>
void ReconData<N>::addTrajComponent(FloatVector *trajComp)
{
    if (m_size != 0 && m_size != trajComp->size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }

    m_size = trajComp->size();
    m_traj1.push_back(std::shared_ptr<FloatVector>(trajComp));
}

template <int N>
void ReconData<N>::setDcf(FloatVector *dcf)
{
    if (m_size != 0 && m_size != dcf->size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }

    m_size = dcf->size();
    m_dcf.reset(dcf);
}

template class ReconData<2>;
template class ReconData<3>;

