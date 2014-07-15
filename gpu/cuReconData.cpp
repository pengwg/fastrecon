#include <iostream>

#include "cuReconData.h"

cuReconData::cuReconData(int size)
    : basicReconData(size)
{
}

void cuReconData::addChannelData(ComplexVector &data)
{
    if (m_size != data.size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }
    ComplexVector *store_data = new ComplexVector(std::move(data));
    auto h_data = reinterpret_cast<const std::vector<cuComplexFloat> *>(store_data);
    auto d_data = new cuComplexVector(*h_data);
    m_kDataMultiChannel.push_back(std::unique_ptr<const cuComplexVector>(d_data));
}

void cuReconData::addTrajComponent(FloatVector &trajComp)
{
    if (m_size != trajComp.size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }

    if (rcDim() == 3)
    {
        std::cout << "3 trajectories have been loaded, ignoring additional data." << std::endl;
        return;
    }
    FloatVector *store_traj = new FloatVector(std::move(trajComp));
    auto d_traj = new cuFloatVector(*store_traj);
    m_traj.push_back(std::unique_ptr<cuFloatVector>(d_traj));

    auto bound = std::minmax_element(store_traj->begin(), store_traj->end());
    m_bounds.push_back(std::make_pair(*bound.first, *bound.second));

    std::cout << "Range: " << '(' << *bound.first << ", " << *bound.second << ')' << std::endl;
}

void cuReconData::setDcf(FloatVector &dcf)
{
    if (m_size != dcf.size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }
    FloatVector *store_dcf = new FloatVector(std::move(dcf));
    auto d_dcf = new cuFloatVector(*store_dcf);
    m_dcf.reset(d_dcf);
}

void cuReconData::transformTrajComponent(float translation, float scale, int comp)
{
    if (comp > rcDim())
    {
        std::cout << "Scale component not exists" << std::endl;
        return;
    }

    for (auto sample : *m_traj[comp])
    {
        sample = (sample + translation) * scale;
    }

    //thrust::transform(m_traj[comp]->begin(), m_traj[comp]->end(), m_traj[comp]->begin(), scale_functor(translation, scale));

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
