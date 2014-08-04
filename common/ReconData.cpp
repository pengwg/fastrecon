#include <iostream>

#include "ReconData.h"

template<template<typename, typename> class C, typename T>
ReconData<C, T>::ReconData(int size)
    : basicReconData<T>(size)
{
}

template<template<typename, typename> class C, typename T>
void ReconData<C, T>::addData(ComplexVector &data)
{
    typedef std::vector<typename LocalComplexVector::value_type> interm_type;
    interm_type &h_data = reinterpret_cast<interm_type &>(data);

    auto d_data = toLocalVector<interm_type, LocalComplexVector>(h_data);
    m_kDataMultiChannel.push_back(std::unique_ptr<const LocalComplexVector>(d_data));
}

template<template<typename, typename> class C, typename T>
void ReconData<C, T>::addTraj(TrajVector &traj)
{
    auto d_traj = toLocalVector<TrajVector, LocalTrajVector>(traj);
    //m_traj.push_back(std::unique_ptr<LocalVector>(d_traj));
    m_traj.reset(d_traj);
}

template<template<typename, typename> class C, typename T>
void ReconData<C, T>::addDcf(Vector &dcf)
{
    auto d_dcf = toLocalVector<Vector, LocalVector>(dcf);
    m_dcf.reset(d_dcf);
}

template<template<typename, typename> class C, typename T>
void ReconData<C, T>::clear()
{
    m_size = 0;
    m_traj.reset();
    m_dcf.reset();
    m_kDataMultiChannel.clear();
    m_bounds.clear();
}

template<template<typename, typename> class C, typename T>
template<typename V, typename LV>
LV *ReconData<C, T>::toLocalVector(V &v) const
{
    V *p = new V(std::move(v));
    if (std::is_same<V, LV>::value)
    {
        return (LV *)p;
    }
    else
    {
        LV *l_p = new LV(*p);
        delete p;
        return l_p;
    }
}

template class ReconData<std::vector, float>;
template class ReconData<thrust::device_vector, float>;
