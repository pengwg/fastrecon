#ifndef IMAGERECON_H
#define IMAGERECON_H

#include "ImageData.h"

class ImageRecon
{
public:
    ImageRecon(const ImageData<float> &imageData, const ImageSize &reconSize);
    ImageData<float> SOS() const;
    ImageData<float> SOS(const ImageData<float> &map) const;

private:
    const ImageData<float> &m_imageData;
    ImageSize m_reconSize;
};

#endif // IMAGERECON_H
