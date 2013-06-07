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

class ReconData
{
public:
    ReconData();
    int dataSize() const;
    int channels() const;

    void setTraj(Traj2D *traj2D);
    void setTraj(Traj3D *traj3D);

    void addChannelData(KData *data);

    const Traj2D *getTraj2D() const;
    const Traj3D *getTraj3D() const;

    const KData *getChannelData(int channel) const;

    void clear();

private:
    int m_rcDim = 0;
    int m_size = 0;

    std::vector<std::shared_ptr<KData> > m_kDataMultiChannel;
    std::shared_ptr<Traj2D> m_traj2D;
    std::shared_ptr<Traj3D> m_traj3D;
};

#endif // RECONDATA_H
