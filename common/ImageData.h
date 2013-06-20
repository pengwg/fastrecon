#ifndef IMAGEDATA_H
#define IMAGEDATA_H

#include <memory>
#include <complex>
#include <vector>

typedef struct
{
    int x;
    int y;
    int z;
} ImageSize;

typedef std::vector<float> FloatVector;
typedef std::vector<std::complex<float> > ComplexVector;

class ImageData
{
public:
    ImageData();
    ImageData(const int dim, const ImageSize &imageSize, ComplexVector *image = nullptr);

    // Shallow copy by default
    ImageData &operator=(const ImageData &imageData) = default;
    // Deep copy
    ImageData makeCopy() const;

    void addChannelImage(ComplexVector *image);
    const ComplexVector *getChannelImage(int channel) const;
    ComplexVector *getChannelImage(int channel);
    int channels() const;
    ImageSize imageSize() const;
    int dataSize() const;
    int dim() const;

    void fftShift();
    void lowFilter(int res);
    void normalize();

private:
    int m_dim;
    ImageSize m_size;
    std::vector<std::shared_ptr<ComplexVector>> m_data;

    void fftShift2(ComplexVector *data);
    void fftShift3(ComplexVector *data);
};

#endif // IMAGEDATA_H