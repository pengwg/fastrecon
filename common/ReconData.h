#ifndef RECONDATA_H
#define RECONDATA_H

#include <memory>
#include "basicReconData.h"

template<typename T>
class ReconData : public basicReconData<T>
{
public:
    using typename basicReconData<T>::Vector;
    using typename basicReconData<T>::ComplexVector;

    ReconData(int size);
    void transformTrajComponent(float translation, float scale, int comp);

    const Vector *getTrajComponent(int comp) const
    { return m_traj[comp].get(); }

    const Vector *getDcf() const
    { return m_dcf.get(); }

    const ComplexVector *getChannelData(int channel) const
    { return m_kDataMultiChannel[channel].get(); }

    int channels() const {return m_kDataMultiChannel.size();}
    int rcDim() const { return m_traj.size(); }
    void clear();

private:
    using basicReconData<T>::m_bounds;
    using basicReconData<T>::m_size;

    virtual void addData(ComplexVector &data) override;
    virtual void addTraj(Vector &traj) override;
    virtual void addDcf(Vector &dcf) override;

    std::vector<std::unique_ptr<const ComplexVector> > m_kDataMultiChannel;
    std::vector<std::unique_ptr<Vector> > m_traj;
    std::unique_ptr<Vector> m_dcf;
};

#endif // RECONDATA_H
