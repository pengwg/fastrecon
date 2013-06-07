#include <iostream>

#include "ReconData.h"

ReconData::ReconData()
{
}

void ReconData::setTraj(Traj2D *traj2D)
{
    m_rcDim = 2;
    if (channels() > 0 && m_size != traj2D->size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_size = traj2D->size();
    m_traj2D.reset(traj2D);
}

void ReconData::setTraj(Traj3D *traj3D)
{
    m_rcDim = 3;
    if (channels() > 0 && m_size != traj3D->size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_size = traj3D->size();
    m_traj3D.reset(traj3D);
}

void ReconData::addChannelData(KData *data)
{
    if (m_size != 0 && m_size != data->size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_size = data->size();
    m_kDataMultiChannel.push_back(std::shared_ptr<KData>(data));
}

const Traj2D *ReconData::getTraj2D() const
{
    return m_traj2D.get();
}

const Traj3D *ReconData::getTraj3D() const
{
    return m_traj3D.get();
}

const KData *ReconData::getChannelData(int channel) const
{
    return m_kDataMultiChannel[channel].get();
}

void ReconData::clear()
{
    m_size = 0;

    m_traj2D.reset();
    m_traj3D.reset();
    m_kDataMultiChannel.clear();
}


int ReconData::dataSize() const
{
    return m_size;
}

int ReconData::channels() const
{
    return m_kDataMultiChannel.size();
}
