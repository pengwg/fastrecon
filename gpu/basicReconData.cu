#include <thrust/transform.h>

#include "basicReconData.h"

template<typename T>
struct scale_functor
{
    const T _a, _b;
    int _dim;
    scale_functor(T a, T b, int dim) : _a(a), _b(b), _dim(dim) {}
    __host__ __device__
    Point<T> operator() (const Point<T> &p) const {
        Point<T> p0;
        for( int d = 0; d < _dim; ++d) {
            p0.x[d] = (p.x[d] + _a) * _b;
        }
        return p0;
    }
};

template<typename T>
void basicReconData<T>::cuScale(thrust::device_vector<Point<T> > &traj, T translation, T scale) const
{
    thrust::transform(traj.begin(), traj.end(), traj.begin(), scale_functor<T>(translation, scale, m_dim));
}

template void basicReconData<float>::cuScale(thrust::device_vector<Point<float> >&, float, float) const;

