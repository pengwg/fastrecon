#ifndef BASICRECONDATA_H
#define BASICRECONDATA_H

#include <complex>
#include <vector>

#include <QStringList>
#include <QFile>

#include <thrust/device_vector.h>

template<typename T>
struct Point {
    T x[4];
};

template<typename T>
class basicReconData
{
public:
    typedef std::vector<T> Vector;
    typedef std::vector<Point<T> > TrajVector;
    typedef std::vector<std::complex<T> > ComplexVector;

    void addChannelData(ComplexVector &data);
    void storeTrajComponent(TrajVector &traj, const Vector &traj_c);
    void setDcf(Vector &dcf);

    void transformTraj(float translation, float scale);
    void loadFromFiles(const QStringList &dataFileList, const QStringList &trajFileList, const QString &dcfFileName);

//  virtual const FloatVector *getTrajComponent(int comp) const = 0;

    std::pair<T, T> getCompBounds(int comp) const {
        return m_bounds[comp];
    }

    int dataSize() const {
        return m_size;
    }
    virtual int channels() const = 0;
    int rcDim() const { return m_dim; }
    virtual void clear() = 0;

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

    struct compute_num_cells_per_sample
    {
        T _half_W;
        int _dim;
        __host__ __device__
        compute_num_cells_per_sample(T half_W, int dim) : _half_W(half_W), _dim(dim) {}

        __host__ __device__
        int operator()(const Point<T> &p) const
        {
            int num_cells = 1;
            for( int d = 0; d < _dim; ++d) {
                int upper_limit = (int)floor((float)p.x[d] + _half_W);
                int lower_limit = (int)ceil((float)p.x[d] - _half_W);
                num_cells *= upper_limit - lower_limit + 1;
            }
            return num_cells;
        }
    };

protected:
    basicReconData(int size);
    virtual ~basicReconData() {}

    virtual void addData(ComplexVector &data) = 0;
    virtual void addTraj(TrajVector &traj) = 0;
    virtual void addDcf(Vector &dcf) = 0;
    virtual void transformLocalTraj(float translation, float scale) = 0;

    void thrust_scale(thrust::device_vector<Point<T> > &traj, T translation, T scale) const;
    void cuComputeCellsPerSample(const thrust::device_vector<Point<T> > &traj, T half_W, thrust::device_vector<int> &cell_coverage) const;

    int m_size;
    int m_dim;
    std::vector<std::pair<T, T> > m_bounds;
};

#endif // BASICRECONDATA_H
