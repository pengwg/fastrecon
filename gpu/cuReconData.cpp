#include <iostream>
#include <thrust/transform.h>
#include "cuReconData.h"

cuReconData::cuReconData(int size)
    : basicReconData(size)
{
}

void cuReconData::addChannelData(const ComplexVector *data)
{
    if (m_size != data->size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }
    auto h_data = reinterpret_cast<const std::vector<cuComplexFloat> *>(data);
    auto d_data = new cuComplexVector(*h_data);
    m_kDataMultiChannel.push_back(std::unique_ptr<const cuComplexVector>(d_data));
}

void cuReconData::addTrajComponent(FloatVector *trajComp)
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
    auto d_trajComp = new cuFloatVector(*trajComp);
    m_traj.push_back(std::unique_ptr<cuFloatVector>(d_trajComp));

    auto bound = std::minmax_element(trajComp->begin(), trajComp->end());
    m_bounds.push_back(std::make_pair(*bound.first, *bound.second));

    std::cout << "Range: " << '(' << *bound.first << ", " << *bound.second << ')' << std::endl;
}

void cuReconData::setDcf(FloatVector *dcf)
{
    if (m_size != dcf->size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }
    auto d_dcf = new cuFloatVector(*dcf);
    m_dcf.reset(d_dcf);
}

void cuReconData::transformTrajComponent(float translation, float scale, int comp)
{
    if (comp > rcDim())
    {
        std::cout << "Scale component not exists" << std::endl;
        return;
    }

    thrust_scale(m_traj[comp].get(), translation, scale);

    m_bounds[comp].first = (m_bounds[comp].first + translation) * scale;
    m_bounds[comp].second = (m_bounds[comp].second + translation) * scale;
}

void cuReconData::clear()
{
    m_size = 0;

    m_traj.clear();
    m_dcf.reset();
    m_kDataMultiChannel.clear();
    m_bounds.clear();
}
