#include <iostream>
#include <thrust/transform.h>
#include "cuReconData.h"

cuReconData::cuReconData(int size)
    : basicReconData(size)
{
}

void cuReconData::addData(ComplexVector &data)
{
    ComplexVector *store_data = new ComplexVector(std::move(data));
    auto h_data = reinterpret_cast<const std::vector<cuComplexFloat> *>(store_data);
    auto d_data = new cuComplexVector(*h_data);
    m_kDataMultiChannel.push_back(std::unique_ptr<const cuComplexVector>(d_data));
}

void cuReconData::addTraj(FloatVector &traj)
{
    FloatVector *store_traj = new FloatVector(std::move(traj));
    auto d_traj = new cuFloatVector(*store_traj);
    m_traj.push_back(std::unique_ptr<cuFloatVector>(d_traj));
}

void cuReconData::addDcf(FloatVector &dcf)
{
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
