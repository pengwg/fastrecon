#ifndef IMAGERECON_H
#define IMAGERECON_H

#include "ImageData.h"

class ImageRecon
{
public:
    ImageRecon(const ImageData<std::vector, float> &imageData, const ImageSize &reconSize);
    ImageData<std::vector, float> SOS() const;
    ImageData<std::vector, float> SOS(const ImageData<std::vector, float> &map) const;

private:
    const ImageData<std::vector, float> &m_imageData;
    ImageSize m_reconSize;
};

#endif // IMAGERECON_H
