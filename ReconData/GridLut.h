#ifndef GRIDLUT_H
#define GRIDLUT_H

#include "ReconData.h"
#include "ImageData.h"

class ConvKernel;

template<typename T>
class GridLut
{
protected:
    GridLut(ReconData<T> &reconData);

public:
    virtual ~GridLut();
    static std::shared_ptr<GridLut<T>> Create(ReconData<T> &reconData);

    virtual void plan(unsigned reconSize, float overGridFactor, float kWidth, unsigned klength = 32);
    virtual std::shared_ptr<ImageData<T>> execute();
    void setNumOfThreads(unsigned threads) {
        m_num_threads = threads;
    }
    void setIndex(int index) {
        m_index = index;
    }

protected:
    ComplexVector<T> griddingChannel(int channel);

    ReconData<T> &m_associatedData;
    unsigned m_dim = 0;
    unsigned m_gridSize = 0;
    ConvKernel *m_kernel = nullptr;
    std::vector<float> m_center[3];
    std::vector<int> m_start[3];
    std::vector<int> m_end[3];

    unsigned m_num_threads = 1; // number of child threads for openmp
    int m_index = 0;
};

#endif // GRIDLUT_H
