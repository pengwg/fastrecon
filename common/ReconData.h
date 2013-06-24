#ifndef RECONDATA_H
#define RECONDATA_H

#include <memory>
#include <QStringList>
#include <QFile>
#include "ImageData.h"

class ReconData
{
public:
    ReconData(int size);

    int dataSize() const {return m_size;}
    int channels() const {return m_kDataMultiChannel.size();}

    void addChannelData(const ComplexVector *data);
    void addTrajComponent(FloatVector *trajComp);
    void setDcf(FloatVector *dcf);
    void loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName);

    const FloatVector *getTrajComponent(int comp) const
        { return m_traj[comp].get(); }

    const FloatVector *getDcf() const
        { return m_dcf.get(); }

    const ComplexVector *getChannelData(int channel) const
        { return m_kDataMultiChannel[channel].get(); }

    int rcDim() const { return m_traj.size(); }
    void clear();

private:
    int m_size;

    std::vector<std::shared_ptr<const ComplexVector> > m_kDataMultiChannel;
    std::vector<std::shared_ptr<FloatVector> > m_traj;
    std::shared_ptr<FloatVector> m_dcf;
};

#endif // RECONDATA_H
