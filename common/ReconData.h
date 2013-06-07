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

typedef std::complex<float> KData;

class ReconData
{
public:
    ReconData();
    int dataSize() const;
    int channels() const;

    void setTraj(KPoint2D *traj2D, int size);
    void setTraj(KPoint3D *traj3D, int size);

    void addChannelData(KData *data, int size);

    const KPoint2D *getTraj2D() const;
    const KPoint3D *getTraj3D() const;

    const KData *getChannelData(int channel) const;

    void clear();

private:
    int m_rcDim = 0;
    int m_size = 0;

    std::vector<std::shared_ptr<KData> > m_kDataMultiChannel;
    std::shared_ptr<KPoint2D> m_kTraj2D;
    std::shared_ptr<KPoint3D> m_kTraj3D;
};

#endif // RECONDATA_H
