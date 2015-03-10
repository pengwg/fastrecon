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
protected:
    ReconData(int samples, int acquisitions);

public:
    typedef std::vector<Point<T>> TrajVector;
    static std::shared_ptr<ReconData<T>> Create(int samples, int acquisitions, bool gpu = false);
    virtual ~ReconData() {}

    void loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName);
    void addChannelData(ComplexVector<T> &&data);
    void setDcf(std::vector<T> &dcf);
    void loadTraj(const QVector<T> &traj, int dim);
    virtual void updateSingleAcquisition(const std::complex<T> *data, int acquisition, int channel = 0);
    void normalizeTraj(unsigned size);    
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
    int channels() const { return m_kDataMultiChannel.size(); }
    virtual void clear();

protected:
    int m_samples;
    int m_acquisitions;
    std::size_t m_size;
    int m_dim = 0;
    std::vector<std::pair<T, T>> m_bounds;

    TrajVector m_traj;
    std::vector<ComplexVector<T>> m_kDataMultiChannel;
    std::vector<T> m_dcf;

private:
    virtual void transformLocalTraj(T translation, T scale);
    void storeTrajComponent(const std::vector<T> &traj_c);
};
#endif // RECONDATA_H
