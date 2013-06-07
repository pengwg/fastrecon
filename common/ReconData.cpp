#include <iostream>

#include "ReconData.h"

ReconData::ReconData()
{
}

void ReconData::setTraj(KPoint2D *traj2D, int size)
{
    m_rcDim = 2;
    if (channels() > 0 && m_size != size)
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_size = size;
    m_kTraj2D.reset(traj2D);
}

void ReconData::setTraj(KPoint3D *traj3D, int size)
{
    m_rcDim = 3;
    if (channels() > 0 && m_size != size)
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_size = size;
    m_kTraj3D.reset(traj3D);
}

void ReconData::addChannelData(KData *data, int size)
{
    if (m_size != 0 && m_size != size)
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_size = size;
    m_kDataMultiChannel.push_back(std::shared_ptr<KData>(data));
}

const KPoint2D *ReconData::getTraj2D() const
{
    return m_kTraj2D.get();
}

const KPoint3D *ReconData::getTraj3D() const
{
    return m_kTraj3D.get();
}

const KData *ReconData::getChannelData(int channel) const
{
    return m_kDataMultiChannel[channel].get();
}

void ReconData::clear()
{
    m_size = 0;

    m_kTraj2D.reset();
    m_kTraj3D.reset();
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
