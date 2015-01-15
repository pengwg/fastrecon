#ifndef RECONDATA_H
#define RECONDATA_H

#include <complex>
#include <vector>
#include <memory>
#include <QStringList>
#include <QFile>

#include "common.h"

template<typename T>
struct Point {
    T x[4];
};

template<typename T>
class ReconData
{
public:
    typedef std::vector<Point<T>> TrajVector;

    ReconData(int samples, int acquisitions);
    virtual ~ReconData() {}

    void addChannelData(const ComplexVector<T> &data);
    void setDcf(std::vector<T> &dcf);

    void transformTraj(T translation, T scale);
    void loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName);
    void loadTraj(const QVector<T> &traj, int dim);
    void updateSingleAcquisition(const std::complex<T> *data, int acquisition, int channel = 0);

    std::pair<T, T> getCompBounds(int comp) const {
        return m_bounds[comp];
    }

    int dataSize() const {
        return m_size;
    }

    int rcDim() const { return m_dim; }

    const TrajVector &getTraj() const {
        return m_traj;
    }

    const std::vector<T> &getDcf() const {
        return m_dcf;
    }

    const ComplexVector<T> *getChannelData(int channel) const;

    virtual int channels() const { return m_kDataMultiChannel.size(); }
    virtual void clear();

protected:
    virtual void transformLocalTraj(T translation, T scale);

    int m_samples;
    int m_acquisitions;
    int m_size;
    int m_dim = 0;
    std::vector<std::pair<T, T>> m_bounds;

    TrajVector m_traj;
    std::vector<std::unique_ptr<ComplexVector<T>>> m_kDataMultiChannel;
    std::vector<T> m_dcf;

private:
    void storeTrajComponent(const std::vector<T> &traj_c);
};
#endif // RECONDATA_H
