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

private:
    int m_dataSize;
    int m_channels;

    std::vector<std::shared_ptr<KData>> m_kDataMultiChannel;
    std::shared_ptr<KTraj> m_kTraj;
};

#endif // RECONDATA_H
