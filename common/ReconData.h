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
} KPoint;

typedef std::vector<KPoint> KTraj;
typedef std::vector<std::complex<float>> KData;

class ReconData
{
public:
    ReconData();
    int dataSize() const;
    int channels() const;

    void setTraj(KTraj *traj);
    void addChannelData(KData *data);

    const KTraj *getTraj() const;
    const KData *getChannelData(int channel) const;

    void clear();

private:
    int m_dataSize = 0;

    std::vector<std::shared_ptr<KData>> m_kDataMultiChannel;
    std::shared_ptr<KTraj> m_kTraj;
};

#endif // RECONDATA_H
