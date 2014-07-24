#ifndef BASICRECONDATA_H
#define BASICRECONDATA_H

#include <complex>
#include <vector>

#include <QStringList>
#include <QFile>

template<typename T>
class basicReconData
{
public:
    typedef std::vector<T> Vector;
    typedef std::vector<std::complex<T>> ComplexVector;

    void addChannelData(ComplexVector &data);
    void addTrajComponent(Vector &trajComp);
    void setDcf(Vector &dcf);

    void transformTrajComponent(float translation, float scale, int comp);
    void loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName);

//  virtual const FloatVector *getTrajComponent(int comp) const = 0;

    std::pair<T, T> getCompBounds(int comp) const {
        return m_bounds[comp];
    }

    int dataSize() const {
        return m_size;
    }
    virtual int channels() const = 0;
    virtual int rcDim() const = 0;
    virtual void clear() = 0;

protected:
    basicReconData(int size);
    virtual ~basicReconData() {}

    virtual void addData(ComplexVector &data) = 0;
    virtual void addTraj(Vector &traj) = 0;
    virtual void addDcf(Vector &dcf) = 0;
    virtual void transformLocalTrajComp(float translation, float scale, int comp) = 0;

    int m_size;
    std::vector<std::pair<T, T>> m_bounds;
};

#endif // BASICRECONDATA_H
