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
    ImageData(const int dim, const ImageSize &size);

    void addChannelImage(ComplexVector *image);
    ComplexVector *getChannelImage(int channel) const;
    int channels() const;
    ImageSize size() const;
    int length() const;
    int dim() const;

    void fftShift();
    ImageData crop_sos(ImageSize size) const;

private:
    int m_dim;
    ImageSize m_size;
    std::vector<std::shared_ptr<ComplexVector>> m_data;

    void fftShift2(ComplexVector *data);
    void fftShift3(ComplexVector *data);
};

#endif // IMAGEDATA_H
