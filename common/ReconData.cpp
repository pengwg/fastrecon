#include <iostream>

#include "ReconData.h"

ReconData::ReconData(int size)
    :m_size(size)
{
}

void ReconData::addChannelData(const ComplexVector *data)
{
    if (m_size != data->size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    m_kDataMultiChannel.push_back(std::shared_ptr<const ComplexVector>(data));
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
    m_traj.push_back(std::shared_ptr<FloatVector>(trajComp));

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

void ReconData::scaleTrajComponent(float lbound, float ubound, int comp)
{
}

void ReconData::loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName)
{
    std::cout << std::endl << "Read trajectory:" << std::endl;
    for (const QString &name : trajFileList)
    {
        std::cout << name.toStdString() << std::endl;
        FloatVector *traj = new FloatVector(m_size);

        QFile file(name);
        file.open(QIODevice::ReadOnly);
        auto count = file.read((char *)traj->data(), m_size * sizeof(FloatVector::value_type));
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
    FloatVector *dcf = new FloatVector(m_size);

    QFile file(dcfFileName);
    file.open(QIODevice::ReadOnly);
    auto count = file.read((char *)dcf->data(), m_size * sizeof(FloatVector::value_type));
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
        ComplexVector *kdata = new ComplexVector(m_size);
        std::cout << name.toStdString() << std::endl;

        QFile file(name);
        file.open(QIODevice::ReadOnly);
        auto count = file.read((char *)kdata->data(), m_size * sizeof(ComplexVector::value_type));
        file.close();

        if (count != m_size * sizeof(ComplexVector::value_type))
        {
            std::cout << "Error: wrong data size in " << name.toStdString() << std::endl;
            std::exit(1);
        }

        addChannelData(kdata);
    }
}

void ReconData::clear()
{
    m_size = 0;

    m_traj.clear();
    m_dcf.reset();
    m_kDataMultiChannel.clear();
    m_bounds.clear();
}
