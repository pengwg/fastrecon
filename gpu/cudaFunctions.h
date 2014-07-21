#ifndef CUDAFUNCTIONS_H
#define CUDAFUNCTIONS_H

template<typename T>
void thrust_scale(thrust::device_vector<T> &traj, T translation, T scale);

#endif
