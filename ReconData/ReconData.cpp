#include <iostream>

#include "ReconData.h"

template<typename T>
ReconData<T>::ReconData(int size) : m_size(size)
{
}

template<typename T>
void ReconData<T>::addChannelData(ComplexVector<T> &data)
{
    if (m_size != data.size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    auto p_data = new ComplexVector<T>(std::move(data));
    m_kDataMultiChannel.push_back(std::unique_ptr<const ComplexVector<T>>(p_data));
}

template<typename T>
void ReconData<T>::setDcf(std::vector<T> &dcf)
{
    if (m_size != dcf.size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }

    auto p_dcf = new std::vector<T>(std::move(dcf));
    m_dcf.reset(p_dcf);
}

template<typename T>
void ReconData<T>::storeTrajComponent(TrajVector &traj, const std::vector<T> &traj_c)
{
    if (m_size != traj_c.size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }

    if (m_dim == 4)
    {
        std::cout << "4-D trajectories have been loaded, ignoring additional data." << std::endl;
        return;
    }

    auto bound = std::minmax_element(traj_c.begin(), traj_c.end());
    m_bounds.push_back(std::make_pair(*bound.first, *bound.second));
    std::cout << "Range: " << '(' << *bound.first << ", " << *bound.second << ')' << std::endl;

    auto in = traj_c.begin();
    for (auto &p : traj)
    {
        p.x[m_dim] = *(in++);
    }
    ++m_dim;
}

template<typename T>
void ReconData<T>::loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName)
{
    std::cout << std::endl << "Read trajectory:" << std::endl;
    auto traj = new TrajVector(m_size);

    for (const QString &name : trajFileList)
    {
        std::cout << name.toStdString() << std::endl;
        std::vector<T> traj_c(m_size);

        QFile file(name);
        file.open(QIODevice::ReadOnly);
        auto count = file.read((char *)traj_c.data(), m_size * sizeof(T));
        file.close();

        if (count != m_size * sizeof(T))
        {
            std::cout << "Error: wrong data size in " << name.toStdString() << std::endl;
            std::exit(1);
        }

        storeTrajComponent(*traj, traj_c);
    }

    m_traj.reset(traj);

    std::cout << std::endl << "Read dcf:" << std::endl;
    std::cout << dcfFileName.toStdString() << std::endl;
    auto dcf = new std::vector<T>(m_size);

    QFile file(dcfFileName);
    file.open(QIODevice::ReadOnly);
    auto count = file.read((char *)dcf->data(), m_size * sizeof(T));
    file.close();

    if (count != m_size * sizeof(T))
    {
        std::cout << "Error: wrong data size in " << dcfFileName.toStdString() << std::endl;
        std::exit(1);
    }
    m_dcf.reset(dcf);

    // Load data
    std::cout << std::endl << "Read data:" << std::endl;
    for (const QString &name : dataFileList)
    {
        ComplexVector<T> kdata(m_size);
        std::cout << name.toStdString() << std::endl;

        QFile file(name);
        file.open(QIODevice::ReadOnly);
        auto count = file.read((char *)kdata.data(), m_size * sizeof(typename ComplexVector<T>::value_type));
        file.close();

        if (count != m_size * sizeof(typename ComplexVector<T>::value_type))
        {
            std::cout << "Error: wrong data size in " << name.toStdString() << std::endl;
            std::exit(1);
        }

        addChannelData(kdata);
    }
}

template<typename T>
void ReconData<T>::transformTraj(T translation, T scale)
{
    transformLocalTraj(translation, scale);

    for (int comp = 0; comp < m_dim; ++comp)
    {
        m_bounds[comp].first = (m_bounds[comp].first + translation) * scale;
        m_bounds[comp].second = (m_bounds[comp].second + translation) * scale;
    }
}

template<typename T>
void ReconData<T>::transformLocalTraj(T translation, T scale)
{
    for (auto &sample : *m_traj)
    {
        for (int comp = 0; comp < m_dim; ++comp)
            sample.x[comp] = (sample.x[comp] + translation) * scale;
    }
}

template<typename T>
void ReconData<T>::clear()
{
    m_size = 0;
    m_traj.reset();
    m_dcf.reset();
    m_kDataMultiChannel.clear();
    m_bounds.clear();
}

template class ReconData<float>;

