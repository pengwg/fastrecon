#ifndef BASICRECONDATA_H
#define BASICRECONDATA_H

#include <complex>
#include <vector>

#include <QStringList>
#include <QFile>

typedef std::vector<float> FloatVector;
typedef std::vector<std::complex<float> > ComplexVector;

class basicReconData
{
public:
    basicReconData(int size);

    virtual void addChannelData(const ComplexVector *data) = 0;
    virtual void addTrajComponent(FloatVector *trajComp) = 0;
    virtual void setDcf(FloatVector *dcf) = 0;
    virtual void transformTrajComponent(float translation, float scale, int comp) = 0;
    void loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName);

//  virtual const FloatVector *getTrajComponent(int comp) const = 0;

//  virtual const FloatVector *getDcf() const
//    { return m_dcf.get(); }

//  const ComplexVector *getChannelData(int channel) const
//   { return m_kDataMultiChannel[channel].get(); }

    std::pair<float, float> getCompBounds(int comp) const {
        return m_bounds[comp];
    }

    int dataSize() const {
        return m_size;
    }
    virtual int channels() const = 0;
    virtual int rcDim() const = 0;
    virtual void clear() = 0;

protected:
    int m_size;
    std::vector<std::pair<float, float>> m_bounds;
};

#endif // BASICRECONDATA_H
