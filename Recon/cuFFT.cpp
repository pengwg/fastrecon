#include <QElapsedTimer>
#include "cuFFT.h"

cuFFT::cuFFT(int dims, ImageSize size, int sign)
    : m_dim(dims), m_size(size), m_sign(sign)
{
    plan();
}

cuFFT::~cuFFT()
{
    cufftDestroy(m_plan);
}

void cuFFT::plan()
{
    if (m_dim == 2)
        cufftPlan2d(&m_plan, m_size.x, m_size.y, CUFFT_C2C);
    else if (m_dim == 3)
        cufftPlan3d(&m_plan, m_size.x, m_size.y, m_size.z, CUFFT_C2C);
}

void cuFFT::excute(cuImageData<float> &imgData)
{
    QElapsedTimer timer;
    timer.start();

    for (int i = 0; i < imgData.channels(); i++)
    {
        auto d_data = static_cast<cufftComplex *>(thrust::raw_pointer_cast(imgData.cuGetChannelImage(i)->data()));
        cufftExecC2C(m_plan, d_data, d_data, CUFFT_INVERSE);
        imgData.update();
        std::cout << " FFT channel " << i << " | " << timer.restart() << " ms" << std::endl;
    }
}
