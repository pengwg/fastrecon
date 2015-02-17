#include <iostream>
#include <limits>
#include <QVector>

#include "ReconData.h"

template<typename T>
ReconData<T>::ReconData(int samples, int acquisitions) : m_samples(samples), m_acquisitions(acquisitions)
{
    m_size = samples * acquisitions;
}

template<typename T>
void ReconData<T>::addChannelData(const ComplexVector<T> &data)
{
    if (m_size != data.size())
    {
        std::cerr << "Error: trajectory and data have different size!" << std::endl;
        exit(1);
    }

    auto p_data = new ComplexVector<T>(data);
    m_kDataMultiChannel.push_back(std::unique_ptr<ComplexVector<T>>(p_data));
}

template<typename T>
void ReconData<T>::setDcf(std::vector<T> &dcf)
{
    if (m_size != dcf.size())
    {
        std::cerr << "Error: data size does not match!" << std::endl;
        exit(1);
    }
    m_dcf = std::move(dcf);
}

template<typename T>
void ReconData<T>::storeTrajComponent(const std::vector<T> &traj_c)
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
    m_traj.resize(m_size);

    auto bound = std::minmax_element(traj_c.begin(), traj_c.end());
    m_bounds.push_back(std::make_pair(*bound.first, *bound.second));
    std::cout << "Range: " << '(' << *bound.first << ", " << *bound.second << ')' << std::endl;

    auto in = traj_c.begin();
    for (auto &p : m_traj)
    {
        p.x[m_dim] = *(in++);
    }
    ++m_dim;
}

template<typename T>
void ReconData<T>::loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName)
{
    std::cout << std::endl << "Read trajectory:" << std::endl;
    m_dim = 0;

    for (const QString &name : trajFileList)
    {
        std::cout << name.toStdString() << std::endl;
        std::vector<T> traj_c(m_size);

        QFile file(name);
        file.open(QIODevice::ReadOnly);
        auto count = file.read((char *)traj_c.data(), m_size * sizeof(T));
        file.close();

        if ((std::size_t)count != m_size * sizeof(T))
        {
            std::cout << "Error: wrong data size in " << name.toStdString() << std::endl;
            std::exit(1);
        }

        storeTrajComponent(traj_c);
    }

    std::cout << std::endl << "Read dcf:" << std::endl;
    std::cout << dcfFileName.toStdString() << std::endl;
    m_dcf.resize(m_size);

    QFile file(dcfFileName);
    file.open(QIODevice::ReadOnly);
    auto count = file.read((char *)m_dcf.data(), m_size * sizeof(T));
    file.close();

    if ((std::size_t)count != m_size * sizeof(T))
    {
        std::cout << "Error: wrong data size in " << dcfFileName.toStdString() << std::endl;
        std::exit(1);
    }

    // Load data
    std::cout << std::endl << "Read data:" << std::endl;
    ComplexVector<T> kdata(m_size);

    for (const QString &name : dataFileList)
    {
        std::cout << name.toStdString() << std::endl;

        QFile file(name);
        file.open(QIODevice::ReadOnly);
        auto count = file.read((char *)kdata.data(), m_size * sizeof(typename ComplexVector<T>::value_type));
        file.close();

        if ((std::size_t)count != m_size * sizeof(typename ComplexVector<T>::value_type))
        {
            std::cout << "Error: wrong data size in " << name.toStdString() << std::endl;
            std::exit(1);
        }

        addChannelData(kdata);
    }
}

template<typename T>
void ReconData<T>::loadTraj(const QVector<T> &traj, int dim)
{
    if ((std::size_t)traj.size() != m_size * (dim + 1)) {
        std::cerr << "Error: data size does not match!" << std::endl << std::flush;
        return;
    }
    m_dim = dim;
    m_traj.resize(m_size);
    m_dcf.resize(m_size);
    m_bounds.resize(dim);

    for (auto &bound : m_bounds) {
        bound.first = std::numeric_limits<T>::max();
        bound.second = std::numeric_limits<T>::lowest();
    }

    auto in = traj.begin();
    auto it_dcf = m_dcf.begin();
    for (auto &p : m_traj)
    {
        for (auto i = 0; i < dim; i++) {
            p.x[i] = *(in++);

            if (m_bounds[i].first > p.x[i])
                m_bounds[i].first = p.x[i];
            if (m_bounds[i].second < p.x[i])
                m_bounds[i].second = p.x[i];
        }
        *(it_dcf++) = *(in++);
    }
}

template<typename T>
void ReconData<T>::updateSingleAcquisition(const std::complex<T> *data, int acquisition, int channel)
{
    if (channel > (int)m_kDataMultiChannel.size() - 1)
    {
        std::cerr << "Error: channel " << channel << " is empty!" << std::endl << std::flush;
        return;
    }
    auto &channel_data = *m_kDataMultiChannel[channel];
    std::copy_n(data, m_samples, channel_data.begin() + m_samples * acquisition);
}

template<typename T>
const ComplexVector<T> *ReconData<T>::getChannelData(int channel) const
{
    if (channel < 0 || channel > (int)m_kDataMultiChannel.size() - 1)
    {
        return nullptr;
    }
    return m_kDataMultiChannel[channel].get();
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
    for (auto &sample : m_traj)
    {
        for (int comp = 0; comp < m_dim; ++comp)
            sample.x[comp] = (sample.x[comp] + translation) * scale;
    }
}

template<typename T>
void ReconData<T>::clear()
{
    m_size = 0;
    m_traj.clear();
    m_dcf.clear();
    m_kDataMultiChannel.clear();
    m_bounds.clear();
}

template class ReconData<float>;

