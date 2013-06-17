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
        for (const auto &data : *input.get())
        {
            *itOut++ += data * data;
        }
    }

    for (auto &data : *out.get())
    {
        data = std::sqrt(data);
    }

    img.push_back(out);
    return img;
}
