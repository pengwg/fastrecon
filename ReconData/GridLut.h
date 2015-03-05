#ifndef GRIDLUT_H
#define GRIDLUT_H

#include "ConvKernel.h"
#include "ReconData.h"
#include "ImageData.h"

template<typename T>
class GridLut
{
public:
    static std::shared_ptr<GridLut<T>> Create(ReconData<T> &reconData, unsigned gridSize, const ConvKernel &kernel);
    virtual ~GridLut();

    virtual void plan();
    virtual std::shared_ptr<ImageData<T>> execute();
    void setNumOfThreads(unsigned threads) {
        m_num_threads = threads;
    }
    void setIndex(int index) {
        m_index = index;
    }

protected:
    GridLut(ReconData<T> &reconData, const ConvKernel &kernel) : m_associatedData(reconData), m_kernel(kernel) {}
    ComplexVector<T> griddingChannel(int channel);

    ReconData<T> &m_associatedData;
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
