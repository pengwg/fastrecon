#include <iostream>

#include "basicReconData.h"

template<typename T>
basicReconData<T>::basicReconData(int size)
    :m_size(size)
{
}

template<typename T>
void basicReconData<T>::addChannelData(ComplexVector &data)
{
    if (m_size != data.size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    addData(data);
}

template<typename T>
void basicReconData<T>::addTrajComponent(Vector &trajComp)
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

    auto bound = std::minmax_element(trajComp.begin(), trajComp.end());
    m_bounds.push_back(std::make_pair(*bound.first, *bound.second));
    std::cout << "Range: " << '(' << *bound.first << ", " << *bound.second << ')' << std::endl;

    addTraj(trajComp);
}

template<typename T>
void basicReconData<T>::setDcf(Vector &dcf)
{
    if (m_size != dcf.size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }

    addDcf(dcf);
}

template<typename T>
void basicReconData<T>::loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName)
{
    std::cout << std::endl << "Read trajectory:" << std::endl;
    for (const QString &name : trajFileList)
    {
        std::cout << name.toStdString() << std::endl;
        Vector traj(m_size);

        QFile file(name);
        file.open(QIODevice::ReadOnly);
        auto count = file.read((char *)traj.data(), m_size * sizeof(typename Vector::value_type));
        file.close();

        if (count != m_size * sizeof(typename Vector::value_type))
        {
            std::cout << "Error: wrong data size in " << name.toStdString() << std::endl;
            std::exit(1);
        }

        addTrajComponent(traj);
    }

    std::cout << std::endl << "Read dcf:" << std::endl;
    std::cout << dcfFileName.toStdString() << std::endl;
    Vector dcf(m_size);

    QFile file(dcfFileName);
    file.open(QIODevice::ReadOnly);
    auto count = file.read((char *)dcf.data(), m_size * sizeof(typename Vector::value_type));
    file.close();

    if (count != m_size * sizeof(typename Vector::value_type))
    {
        std::cout << "Error: wrong data size in " << dcfFileName.toStdString() << std::endl;
        std::exit(1);
    }
    setDcf(dcf);

    // Load data
    std::cout << std::endl << "Read data:" << std::endl;
    for (const QString &name : dataFileList)
    {
        ComplexVector kdata(m_size);
        std::cout << name.toStdString() << std::endl;

        QFile file(name);
        file.open(QIODevice::ReadOnly);
        auto count = file.read((char *)kdata.data(), m_size * sizeof(typename ComplexVector::value_type));
        file.close();

        if (count != m_size * sizeof(typename ComplexVector::value_type))
        {
            std::cout << "Error: wrong data size in " << name.toStdString() << std::endl;
            std::exit(1);
        }

        addChannelData(kdata);
    }
}

template<typename T>
void basicReconData<T>::transformTrajComponent(float translation, float scale, int comp)
{
    if (comp > rcDim())
    {
        std::cout << "Scale component not exists" << std::endl;
        return;
    }

    transformLocalTrajComp(translation, scale, comp);

    m_bounds[comp].first = (m_bounds[comp].first + translation) * scale;
    m_bounds[comp].second = (m_bounds[comp].second + translation) * scale;
}

template class basicReconData<float>;
