#include <iostream>
#include <omp.h>
#include "SOS.h"

SOS::SOS()
{
}

ImageData SOS::execute(const ImageData &imgData)
{
    ImageData img(imgData.size());
    auto out = new ComplexVector(imgData.length());

    for (int n = 0; n < imgData.channels(); n++)
    {
        auto itOut = out->begin();
        auto itInput = imgData.getChannelImage(n)->cbegin();

#pragma omp parallel for
        for (int i = 0; i < imgData.length(); i++)
        {
            auto data = *(itInput + i);
            *(itOut+i) += data * std::conj(data);
        }
    }

    for (auto &data : *out)
    {
        data = std::sqrt(data);
    }

    img.addChannelImage(out);
    return img;
}
