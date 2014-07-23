#ifndef IMAGERECON_H
#define IMAGERECON_H

#include "hostImageData.h"

class ImageRecon
{
public:
    ImageRecon(const hostImageData<float> &imageData, const ImageSize &reconSize);
    hostImageData<float> SOS() const;
    hostImageData<float> SOS(const ImageData<std::vector, float> &map) const;

private:
    const hostImageData<float> &m_imageData;
    ImageSize m_reconSize;
};

#endif // IMAGERECON_H
