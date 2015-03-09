#include <QElapsedTimer>
#include "cuFFT.h"
#include "cuImageData.h"

cuFFT::cuFFT(int dims, ImageSize size, int sign)
    : FFT(dims, size), m_sign(sign)
{
}

cuFFT::~cuFFT()
{
    cufftDestroy(m_plan);
}

void cuFFT::plan()
{
    std::cout << "Create GPU FFT plan." << std::endl;
    if (m_dim == 2)
        cufftPlan2d(&m_plan, m_size.x, m_size.y, CUFFT_C2C);
    else if (m_dim == 3)
        cufftPlan3d(&m_plan, m_size.x, m_size.y, m_size.z, CUFFT_C2C);
}

void cuFFT::excute(ImageData<float> &imgData)
{
    QElapsedTimer timer;
    timer.start();

    auto &cu_imgData = dynamic_cast<cuImageData<float> &>(imgData);
    for (int i = 0; i < cu_imgData.channels(); i++)
    {
        auto d_data = static_cast<cufftComplex *>(thrust::raw_pointer_cast(cu_imgData.cuGetChannelImage(i)->data()));
        cufftExecC2C(m_plan, d_data, d_data, m_sign);
        cu_imgData.syncDeviceToHost();
        std::cout << "GPU FFT channel " << this->m_index << ':' << i << " | " << timer.restart() << " ms" << std::endl;
    }
}
