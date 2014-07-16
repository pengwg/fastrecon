#include <iostream>

#include "ReconData.h"

template<typename T>
ReconData<T>::ReconData(int size)
    : basicReconData<T>(size)
{
}

template<typename T>
void ReconData<T>::addData(ComplexVector &data)
{
    ComplexVector *store_data = new ComplexVector(std::move(data));
    m_kDataMultiChannel.push_back(std::unique_ptr<const ComplexVector>(store_data));
}

template<typename T>
void ReconData<T>::addTraj(Vector &traj)
{
    Vector *store_traj = new Vector(std::move(traj));
    m_traj.push_back(std::unique_ptr<Vector>(store_traj));
}

template<typename T>
void ReconData<T>::addDcf(Vector &dcf)
{
    Vector *store_dcf = new Vector(std::move(dcf));
    m_dcf.reset(store_dcf);
}

template<typename T>
void ReconData<T>::transformTrajComponent(float translation, float scale, int comp)
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

template<typename T>
void ReconData<T>::clear()
{
    m_size = 0;

    m_traj.clear();
    m_dcf.reset();
    m_kDataMultiChannel.clear();
    m_bounds.clear();
}

template class ReconData<float>;
