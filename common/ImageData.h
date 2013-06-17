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
    ImageData(const ImageSize &size);

    void addChannelImage(ComplexVector *image);
    ComplexVector *getChannelImage(int channel);
    const ComplexVector *getChannelImage(int channel) const;
    int channels() const;
    ImageSize size() const;
    int length() const;

private:
    ImageSize m_size;
    std::vector<std::shared_ptr<ComplexVector>> m_data;
};

#endif // IMAGEDATA_H
