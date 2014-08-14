#ifndef BASICRECONDATA_H
#define BASICRECONDATA_H

#include <complex>
#include <vector>
#include <QStringList>
#include <QFile>
#include <thrust/device_vector.h>

#include "common.h"

template<typename T>
struct Point {
    T x[4];
};

template<typename T>
class basicReconData
{
public:
    typedef std::vector<Point<T>> TrajVector;

    void addChannelData(ComplexVector<T> &data);
    void storeTrajComponent(TrajVector &traj, const std::vector<T> &traj_c);
    void setDcf(std::vector<T> &dcf);

    void transformTraj(T translation, T scale);
    void loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName);

    std::pair<T, T> getCompBounds(int comp) const {
        return m_bounds[comp];
    }

    int dataSize() const {
        return m_size;
    }
    virtual int channels() const = 0;
    int rcDim() const { return m_dim; }
    virtual void clear() = 0;

protected:
    basicReconData(int size);
    virtual ~basicReconData() {}

    virtual void addData(ComplexVector<T> &data) = 0;
    virtual void addTraj(TrajVector &traj) = 0;
    virtual void addDcf(std::vector<T> &dcf) = 0;
    virtual void transformLocalTraj(float translation, float scale) = 0;

    int m_size;
    int m_dim;
    std::vector<std::pair<T, T>> m_bounds;
};

#endif // BASICRECONDATA_H
