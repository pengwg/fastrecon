#include <iostream>
#include <omp.h>
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
        auto itInput = input.get()->begin();

#pragma omp parallel for
        for (int i = 0; i < input.get()->size(); i++)
        {
            auto data = *(itInput + i);
            *(itOut+i) += data * std::conj(data);
        }
    }

    for (auto &data : *out.get())
    {
        data = std::sqrt(data);
    }

    img.push_back(out);
    return img;
}
