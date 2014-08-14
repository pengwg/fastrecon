#include <thrust/transform.h>

#include "cuReconData.h"

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
void cuReconData<T>::transformLocalTraj(T translation, T scale)
{
    auto traj = this->m_traj.get();

    thrust::transform(traj->begin(), traj->end(), traj->begin(), scale_functor<T>(translation, scale, this->rcDim()));
}

template void cuReconData<float>::transformLocalTraj(float, float);