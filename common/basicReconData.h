#ifndef BASICRECONDATA_H
#define BASICRECONDATA_H

#include <complex>
#include <vector>

#include <QStringList>
#include <QFile>

#include <thrust/device_vector.h>

template<typename T>
class basicReconData
{
public:
    typedef std::vector<T> Vector;
    typedef std::vector<std::complex<T> > ComplexVector;

    void addChannelData(ComplexVector &data);
    void addTrajComponent(Vector &trajComp);
    void setDcf(Vector &dcf);

    void transformTrajComponent(float translation, float scale, int comp);
    void loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName);

//  virtual const FloatVector *getTrajComponent(int comp) const = 0;

    std::pair<T, T> getCompBounds(int comp) const {
        return m_bounds[comp];
    }

    int dataSize() const {
        return m_size;
    }
    virtual int channels() const = 0;
    virtual int rcDim() const = 0;
    virtual void clear() = 0;

    struct scale_functor
    {
        const T a, b;
        scale_functor(T _a, T _b) : a(_a), b(_b) {}
        __host__ __device__
            T operator() (const T& x) const {
                return (x + a) * b;
            }
    };

    struct compute_sample_coverage
    {
        T _half_W;
        __host__ __device__
        compute_sample_coverage(T half_W) : _half_W(half_W) {}

        __host__ __device__
        unsigned int operator()(T p) const
        {
            unsigned int upper_limit = (unsigned int)floor((float)p +_half_W);
            unsigned int lower_limit = (unsigned int)ceil((float)p -_half_W);
            return upper_limit - lower_limit + 1;
        }
    };

protected:
    basicReconData(int size);
    virtual ~basicReconData() {}

    virtual void addData(ComplexVector &data) = 0;
    virtual void addTraj(Vector &traj) = 0;
    virtual void addDcf(Vector &dcf) = 0;
    virtual void transformLocalTrajComp(float translation, float scale, int comp) = 0;

    void thrust_scale(thrust::device_vector<T> &traj, T translation, T scale) const;
    void cuComputeSampleCoverage(const thrust::device_vector<T> &traj, T half_W, thrust::device_vector<unsigned> &cell_coverage) const;
    void cuMultiplies(const thrust::device_vector<unsigned> &in1, const thrust::device_vector<unsigned> &in2, thrust::device_vector<unsigned> &out) const;
    int m_size;
    std::vector<std::pair<T, T> > m_bounds;
};

#endif // BASICRECONDATA_H
