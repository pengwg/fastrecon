#ifndef IMAGEDATA_H
#define IMAGEDATA_H

#include <memory>
#include "basicImageData.h"
#include "common.h"

typedef std::vector<float> FloatVector;
typedef std::vector<std::complex<float>> ComplexVector;

class ImageData : public basicImageData
{
public:
    ImageData();
    ImageData(const int dim, const ImageSize &imageSize, ComplexVector *image = nullptr);

    // Copy
    ImageData(const basicImageData &imageData);
    ImageData(const ImageData &imageData);
    // Move
    ImageData(basicImageData &&imageData);
    ImageData(ImageData &&imageData);

    // Copy
    ImageData &operator=(const basicImageData &ImageData);
    ImageData &operator=(const ImageData &ImageData);
    // Move
    ImageData &operator=(basicImageData &&ImageData);
    ImageData &operator=(ImageData &&ImageData);

    void addChannelImage(ComplexVector *image);
    const ComplexVector *getChannelImage(int channel) const;
    ComplexVector *getChannelImage(int channel);

    virtual void fftShift() override;
    virtual void lowFilter(int res) override;
    virtual void normalize() override;

private:
    std::vector<std::unique_ptr<ComplexVector>> m_data;

    void copy(const basicImageData &imageData);
    void move(basicImageData &imageData);

    void fftShift2(ComplexVector *data);
    void fftShift3(ComplexVector *data);
};

#endif // IMAGEDATA_H
