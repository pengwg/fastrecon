#ifndef CURECONDATA_H
#define CURECONDATA_H

#include "ReconData.h"

template<typename T>
class cuReconData : public ReconData<T>
{
public:
    typedef cuVector<Point<T>> cuTrajVector;

    cuReconData(int size);
    virtual ~cuReconData() {}

    const cuComplexVector<T> *cuGetChannelData(int channel);

    cuTrajVector *cuGetTraj();

    const cuVector<T> *cuGetDcf();

    virtual void clear() override;

    template<typename T1>
    struct scale_functor
    {
        const T1 _a, _b;
        int _dim;
        scale_functor(T1 a, T1 b, int dim) : _a(a), _b(b), _dim(dim) {}
        __host__ __device__
        Point<T1> operator() (const Point<T1> &p) const {
            Point<T1> p0;
            for( int d = 0; d < _dim; ++d) {
                p0.x[d] = (p.x[d] + _a) * _b;
            }
            return p0;
        }
    };

private:
    virtual void transformLocalTraj(T translation, T scale) override;

    std::unique_ptr<cuTrajVector> m_cu_traj;
    std::unique_ptr<cuComplexVector<T>> m_cu_kData;
    std::unique_ptr<cuVector<T>> m_cu_dcf;
    int m_channel_in_device = -1;
};

#endif // CURECONDATA_H
