#ifndef IMAGERECON_H
#define IMAGERECON_H

#include "ImageData.h"

class ImageRecon
{
public:
    ImageRecon(ImageData &imageData, const ImageSize &reconSize);
    ImageData SOS() const;
    ImageData SOS(ImageData &map) const;

private:
    ImageData m_imageData;
    ImageSize m_reconSize;
};

#endif // IMAGERECON_H
