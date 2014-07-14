#include <iostream>

#include "basicReconData.h"

basicReconData::basicReconData(int size)
    :m_size(size)
{
}

void basicReconData::loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName)
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
