#include "ImageRecon.h"

ImageRecon::ImageRecon(const ImageData<std::vector, float> &imageData, const ImageSize &reconSize)
    : m_imageData(imageData), m_reconSize(reconSize)
{
}

ImageData<std::vector, float> ImageRecon::SOS() const
{
    ImageData<std::vector, float> img = SOS(m_imageData);

    for (auto &data : *img.getChannelImage(0))
    {
        data = std::sqrt(data);
    }

    return img;
}

ImageData<std::vector, float> ImageRecon::SOS(const ImageData<std::vector, float> &map) const
{
    auto imageSize = m_imageData.imageSize();
    auto x0 = (imageSize.x - m_reconSize.x) / 2;
    auto y0 = (imageSize.y - m_reconSize.y) / 2;
    auto z0 = (imageSize.z - m_reconSize.z) / 2;

    auto out = new ComplexVector(m_reconSize.x * m_reconSize.y * m_reconSize.z);

    for (int n = 0; n < m_imageData.channels(); n++)
    {
        auto itOut = out->begin();
        auto itInput = m_imageData.getChannelImage(n)->cbegin();
        auto itMap = map.getChannelImage(n)->cbegin();

#pragma omp parallel for
        for (int z = 0; z < m_reconSize.z; z++)
        {
            auto in1 = (z + z0) * (imageSize.x * imageSize.y) + y0 * imageSize.x;
            auto out1 = z * (m_reconSize.x * m_reconSize.y);
            for (int y = 0; y < m_reconSize.y; y++)
            {
                auto in2 = y * imageSize.x + in1 + x0;
                auto out2 = y * m_reconSize.x + out1;
                for (int x = 0; x < m_reconSize.x; x++)
                {
                    auto in3 = x + in2;
                    auto out3 = x + out2;

                    auto data = *(itInput + in3);
                    auto mapData = *(itMap + in3);
                    *(itOut+out3) += data * std::conj(mapData);
                }
            }
        }
    }
    return ImageData<std::vector, float>(m_imageData.dim(), m_reconSize, out);;
}
