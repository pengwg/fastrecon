#include "SOS.h"

SOS::SOS()
{
}

ImageData SOS::execute(const ImageData &imgData)
{
    ImageData img;
    auto out = std::make_shared<ComplexVector>(imgData[0].get()->size());

    for (const auto &input : imgData)
    {
        auto itOut = out.get()->begin();

#pragma omp parallel for
        for (auto itData = input.get()->cbegin(); itData < input.get()->cend(); itData++)
        {
            *itOut++ += *itData * *itData;
        }
    }

    for (auto &data : *out.get())
    {
        data = std::sqrt(data);
    }

    img.push_back(out);
    return img;
}
