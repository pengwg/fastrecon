#ifndef RECONDATA_H
#define RECONDATA_H

#include <memory>
#include <complex>
#include <vector>

template <int N>
struct KPoint
{
    float pos[N];
    float dcf;
};

template <int N>
using Traj = std::vector<KPoint<N> >;

typedef std::vector<std::complex<float> > ComplexVector;
typedef std::vector<float> FloatVector;

template <int N>
class ReconData
{
public:
    ReconData() {}

    int dataSize() const {return m_size;}
    int channels() const {return m_kDataMultiChannel.size();}

    void setTraj(Traj<N> *traj);
    void addChannelData(ComplexVector *data);
    void addTrajComponent(FloatVector *trajComp);
    void setDcf(FloatVector *dcf);

    const Traj<N> *getTraj() const { return m_traj.get(); }

    const ComplexVector *getChannelData(int channel) const
    { return m_kDataMultiChannel[channel].get(); }

    void clear()
    {
        m_size = 0;
        m_rcDim = 0;

        m_traj.reset();
        m_kDataMultiChannel.clear();
    }

    int rcDim() const { return m_rcDim; }

private:
    int m_rcDim = 0;
    int m_size = 0;

    std::vector<std::shared_ptr<ComplexVector> > m_kDataMultiChannel;
    std::shared_ptr<Traj<N> > m_traj;
    std::vector<std::shared_ptr<FloatVector> > m_traj1;
    std::shared_ptr<FloatVector> m_dcf;
};

#endif // RECONDATA_H
