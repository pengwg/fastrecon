#include <iostream>

#include "ReconData.h"

ReconData::ReconData(int size)
    : basicReconData(size)
{
}

void ReconData::addData(ComplexVector &data)
{
    ComplexVector *store_data = new ComplexVector(std::move(data));
    m_kDataMultiChannel.push_back(std::unique_ptr<const ComplexVector>(store_data));
}

void ReconData::addTraj(FloatVector &traj)
{
    FloatVector *store_traj = new FloatVector(std::move(traj));
    m_traj.push_back(std::unique_ptr<FloatVector>(store_traj));
}

void ReconData::addDcf(FloatVector &dcf)
{
    FloatVector *store_dcf = new FloatVector(std::move(dcf));
    m_dcf.reset(store_dcf);
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
