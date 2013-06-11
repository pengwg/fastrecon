#include <iostream>

#include "ReconData.h"

void ReconData::addChannelData(ComplexVector *data)
{
    if (m_size != 0 && m_size != data->size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_size = data->size();
    m_kDataMultiChannel.push_back(std::shared_ptr<ComplexVector>(data));
}

void ReconData::addTrajComponent(FloatVector *trajComp)
{
    if (m_size != 0 && m_size != trajComp->size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }

    m_size = trajComp->size();
    m_traj.push_back(std::shared_ptr<FloatVector>(trajComp));
}

void ReconData::setDcf(FloatVector *dcf)
{
    if (m_size != 0 && m_size != dcf->size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }

    m_size = dcf->size();
    m_dcf.reset(dcf);
}

void ReconData::clear()
{
    m_size = 0;

    m_traj.clear();
    m_dcf.reset();
    m_kDataMultiChannel.clear();
}
