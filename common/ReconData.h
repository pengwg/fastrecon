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

typedef std::vector<std::complex<float> > KData;

template <int N>
class ReconData
{
public:
    ReconData() {}

    int dataSize() const {return m_size;}
    int channels() const {return m_kDataMultiChannel.size();}

    void setTraj(Traj<N> *traj);
    void addChannelData(KData *data);

    const Traj<N> *getTraj() const { return m_traj.get(); }

    const KData *getChannelData(int channel) const
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

    std::vector<std::shared_ptr<KData> > m_kDataMultiChannel;
    std::shared_ptr<Traj<N> > m_traj;
};

#endif // RECONDATA_H
