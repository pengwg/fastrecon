#ifndef RECONDATA_H
#define RECONDATA_H

#include <memory>
#include <complex>
#include <vector>

typedef struct
{
    float kx;
    float ky;
    float dcf;
} KPoint2D;

typedef struct
{
    float kx;
    float ky;
    float kz;
    float dcf;
} KPoint3D;

typedef std::vector<KPoint2D> Traj2D;
typedef std::vector<KPoint3D> Traj3D;
typedef std::vector<std::complex<float> > KData;

template <typename T>
class ReconData
{
public:
    ReconData() {}

    int dataSize() const {return m_size;}
    int channels() const {return m_kDataMultiChannel.size();}

    void setTraj(T *traj);
    void addChannelData(KData *data);

    const T *getTraj() const { return m_traj.get(); }

    const KData *getChannelData(int channel) const
    { return m_kDataMultiChannel[channel].get(); }

    void clear()
    {
        m_size = 0;

        m_traj.reset();
        m_kDataMultiChannel.clear();
    }

private:
    int m_rcDim = 0;
    int m_size = 0;

    std::vector<std::shared_ptr<KData> > m_kDataMultiChannel;
    std::shared_ptr<T> m_traj;
};

extern template class ReconData<Traj2D>;
extern template class ReconData<Traj3D>;

#endif // RECONDATA_H
