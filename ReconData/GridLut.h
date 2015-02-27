#ifndef GRIDLUT_H
#define GRIDLUT_H

#include "ConvKernel.h"
#include "ReconData.h"
#include "ImageData.h"

template<typename T>
class GridLut
{
public:
    GridLut(unsigned dim, unsigned gridSize, ConvKernel &kernel);
    virtual ~GridLut();

    virtual void plan(ReconData<T> &reconData);
    virtual std::shared_ptr<ImageData<T>> execute(ReconData<T> &reconData);
    void setNumOfThreads(unsigned threads) {
        m_num_threads = threads;
    }
    void setIndex(int index) {
        m_index = index;
    }

protected:
    std::unique_ptr<ComplexVector<T>> griddingChannel(const ReconData<T> &reconData, int channel);

    unsigned m_dim;
    unsigned m_gridSize;
    ConvKernel m_kernel;
    std::vector<float> m_center[3];
    std::vector<int> m_start[3];
    std::vector<int> m_end[3];

    unsigned m_num_threads = 1; // number of child threads for openmp
    int m_index = 0;
};

#endif // GRIDLUT_H
