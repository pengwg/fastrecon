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
    // Deep copy
    ImageData(const ImageData &imageData);
    // Move data
    ImageData(ImageData &&imageData);
    ImageData(const int dim, const ImageSize &imageSize, ComplexVector *image = nullptr);
    ImageData &operator=(const ImageData &ImageData);
    ImageData &operator=(ImageData &&ImageData);

    void addChannelImage(ComplexVector *image);
    const ComplexVector *getChannelImage(int channel) const;
    ComplexVector *getChannelImage(int channel);

    virtual void fftShift() override;
    virtual void lowFilter(int res) override;
    virtual void normalize() override;

private:
    std::vector<std::unique_ptr<ComplexVector>> m_data;

    virtual void copy(const ImageData &imageData);
    virtual void move(ImageData &imageData);

    void fftShift2(ComplexVector *data);
    void fftShift3(ComplexVector *data);
};

#endif // IMAGEDATA_H
