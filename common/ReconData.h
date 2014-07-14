#ifndef RECONDATA_H
#define RECONDATA_H

#include <memory>
#include "basicReconData.h"

class ReconData : public basicReconData
{
public:
    ReconData(int size);

    virtual void addChannelData(const ComplexVector *data) override;
    virtual void addTrajComponent(FloatVector *trajComp) override;
    virtual void setDcf(FloatVector *dcf) override;
    void transformTrajComponent(float translation, float scale, int comp);

    const FloatVector *getTrajComponent(int comp) const
    { return m_traj[comp].get(); }

    const FloatVector *getDcf() const
    { return m_dcf.get(); }

    const ComplexVector *getChannelData(int channel) const
    { return m_kDataMultiChannel[channel].get(); }

    std::pair<float, float> getCompBounds(int comp) const
    { return m_bounds[comp]; }

    int channels() const {return m_kDataMultiChannel.size();}
    int rcDim() const { return m_traj.size(); }
    void clear();

private:
    std::vector<std::unique_ptr<const ComplexVector> > m_kDataMultiChannel;
    std::vector<std::unique_ptr<FloatVector> > m_traj;
    std::unique_ptr<FloatVector> m_dcf;
};

#endif // RECONDATA_H
