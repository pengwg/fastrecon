#ifndef BASICRECONDATA_H
#define BASICRECONDATA_H

#include <complex>
#include <vector>

#include <QStringList>
#include <QFile>

#include <thrust/device_vector.h>

template<typename T>
struct Point {
    T x[4];
};

template<typename T>
class basicReconData
{
public:
    typedef std::vector<T> Vector;
    typedef std::vector<Point<T> > TrajVector;
    typedef std::vector<std::complex<T> > ComplexVector;

    void addChannelData(ComplexVector &data);
    void storeTrajComponent(TrajVector &traj, const Vector &traj_c);
    void setDcf(Vector &dcf);

    void transformTraj(float translation, float scale);
    void loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName);

//  virtual const FloatVector *getTrajComponent(int comp) const = 0;

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

    virtual void addData(ComplexVector &data) = 0;
    virtual void addTraj(TrajVector &traj) = 0;
    virtual void addDcf(Vector &dcf) = 0;
    virtual void transformLocalTraj(float translation, float scale) = 0;

    void cuScale(thrust::device_vector<Point<T> > &traj, T translation, T scale) const;
    void cuPreprocess(const thrust::device_vector<Point<T> > &traj, int reconSize, T half_W,
                      thrust::host_vector<int> *tuples_last_h,
                      thrust::host_vector<unsigned> *bucket_begin, thrust::host_vector<unsigned> *bucket_end) const;

    int m_size;
    int m_dim;
    std::vector<std::pair<T, T> > m_bounds;
};

#endif // BASICRECONDATA_H
