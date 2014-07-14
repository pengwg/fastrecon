#include <iostream>

#include "ReconData.h"

ReconData::ReconData(int size)
    : basicReconData(size)
{
}

void ReconData::addChannelData(const ComplexVector *data)
{
    if (m_size != data->size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_kDataMultiChannel.push_back(std::unique_ptr<const ComplexVector>(data));
}

void ReconData::addTrajComponent(FloatVector *trajComp)
{
    if (m_size != trajComp->size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }

    if (rcDim() == 3)
    {
        std::cout << "3 trajectories have been loaded, ignoring additional data." << std::endl;
        return;
    }
    m_traj.push_back(std::unique_ptr<FloatVector>(trajComp));

    auto bound = std::minmax_element(trajComp->begin(), trajComp->end());
    m_bounds.push_back(std::make_pair(*bound.first, *bound.second));

    std::cout << "Range: " << '(' << *bound.first << ", " << *bound.second << ')' << std::endl;
}

void ReconData::setDcf(FloatVector *dcf)
{
    if (m_size != dcf->size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }

    m_dcf.reset(dcf);
}

void ReconData::transformTrajComponent(float translation, float scale, int comp)
{
    if (comp > rcDim())
    {
        std::cout << "Scale component not exists" << std::endl;
        return;
    }

    for (auto &sample : *m_traj[comp])
    {
        sample = (sample + translation) * scale;
    }

    m_bounds[comp].first = (m_bounds[comp].first + translation) * scale;
    m_bounds[comp].second = (m_bounds[comp].second + translation) * scale;
}

void ReconData::clear()
{
    m_size = 0;

    m_traj.clear();
    m_dcf.reset();
    m_kDataMultiChannel.clear();
    m_bounds.clear();
}
