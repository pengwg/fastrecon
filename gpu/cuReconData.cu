#include <thrust/transform.h>
#include <thrust/device_vector.h>

typedef thrust::device_vector<float> cuFloatVector;

struct scale_functor
{
    const float a, b;
    scale_functor(float _a, float _b) : a(_a), b(_b) {}
    __host__ __device__
        float operator() (const float& x) const {
            return (x + a) * b;
        }
};

void thrust_scale(cuFloatVector *traj, float translation, float scale)
{
    thrust::transform(traj->begin(), traj->end(), traj->begin(), scale_functor(translation, scale));
}
