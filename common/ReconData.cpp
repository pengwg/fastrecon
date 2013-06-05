#include <iostream>

#include "ReconData.h"

ReconData::ReconData()
{
}

void ReconData::setTraj(KTraj *traj)
{
    if (m_dataSize != 0 && m_dataSize != traj->size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_dataSize = traj->size();
    m_kTraj.reset(traj);
}

void ReconData::addChannelData(KData *data)
{
    if (m_dataSize != 0 && m_dataSize != data->size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_dataSize = data->size();
    m_kDataMultiChannel.push_back(std::shared_ptr<KData>(data));
}

const KTraj *ReconData::getTraj() const
{
    return m_kTraj.get();
}

const KData *ReconData::getChannelData(int channel) const
{
    return m_kDataMultiChannel[channel].get();
}

void ReconData::clear()
{
    m_dataSize = 0;

    m_kTraj.reset();
    m_kDataMultiChannel.clear();
}


int ReconData::dataSize() const
{
    return m_dataSize;
}

int ReconData::channels() const
{
    return m_kDataMultiChannel.size();
}
