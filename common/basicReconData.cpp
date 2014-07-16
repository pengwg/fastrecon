#include <iostream>

#include "basicReconData.h"

basicReconData::basicReconData(int size)
    :m_size(size)
{
}

void basicReconData::addChannelData(ComplexVector &data)
{
    if (m_size != data.size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    addData(data);
}

void basicReconData::addTrajComponent(FloatVector &trajComp)
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

void basicReconData::setDcf(FloatVector &dcf)
{
    if (m_size != dcf.size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }

    addDcf(dcf);
}

void basicReconData::loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName)
{
    std::cout << std::endl << "Read trajectory:" << std::endl;
    for (const QString &name : trajFileList)
    {
        std::cout << name.toStdString() << std::endl;
        FloatVector traj(m_size);

        QFile file(name);
        file.open(QIODevice::ReadOnly);
        auto count = file.read((char *)traj.data(), m_size * sizeof(FloatVector::value_type));
        file.close();

        if (count != m_size * sizeof(FloatVector::value_type))
        {
            std::cout << "Error: wrong data size in " << name.toStdString() << std::endl;
            std::exit(1);
        }

        addTrajComponent(traj);
    }

    std::cout << std::endl << "Read dcf:" << std::endl;
    std::cout << dcfFileName.toStdString() << std::endl;
    FloatVector dcf(m_size);

    QFile file(dcfFileName);
    file.open(QIODevice::ReadOnly);
    auto count = file.read((char *)dcf.data(), m_size * sizeof(FloatVector::value_type));
    file.close();

    if (count != m_size * sizeof(FloatVector::value_type))
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
        auto count = file.read((char *)kdata.data(), m_size * sizeof(ComplexVector::value_type));
        file.close();

        if (count != m_size * sizeof(ComplexVector::value_type))
        {
            std::cout << "Error: wrong data size in " << name.toStdString() << std::endl;
            std::exit(1);
        }

        addChannelData(kdata);
    }
}
