#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include "cudaFunctions.h"

template<typename T>
struct scale_functor
{
    const T a, b;
    scale_functor(T _a, T _b) : a(_a), b(_b) {}
    __host__ __device__
        T operator() (const T& x) const {
            return (x + a) * b;
        }
};

template<typename T>
void thrust_scale(thrust::device_vector<T> &traj, T translation, T scale)
{
    thrust::transform(traj.begin(), traj.end(), traj.begin(), scale_functor<T>(translation, scale));
}

template void thrust_scale(thrust::device_vector<float> &traj, float translation, float scale);
