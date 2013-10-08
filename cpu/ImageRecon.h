#ifndef IMAGERECON_H
#define IMAGERECON_H

#include "ImageData.h"

class ImageRecon
{
public:
    ImageRecon(const ImageData &imageData, const ImageSize &reconSize);
    ImageData SOS() const;
    ImageData SOS(const ImageData &map) const;

private:
    const ImageData &m_imageData;
    ImageSize m_reconSize;
};

#endif // IMAGERECON_H
