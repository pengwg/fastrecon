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
    // Deep copy
    ImageData(const ImageData &imageData);
    // Move data
    ImageData(ImageData &&imageData);
    ImageData(const int dim, const ImageSize &imageSize, ComplexVector *image = nullptr);

    ImageData &operator=(const ImageData &imageData);
    ImageData &operator=(ImageData &&imageData);


    void addChannelImage(ComplexVector *image);
    const ComplexVector *getChannelImage(int channel) const;
    ComplexVector *getChannelImage(int channel);
    int channels() const { return m_data.size(); }
    ImageSize imageSize() const { return m_size; }
    int dataSize() const;
    int dim() const { return m_dim; }

    void fftShift();
    void lowFilter(int res);
    void normalize();

private:
    int m_dim;
    ImageSize m_size;
    std::vector<std::unique_ptr<ComplexVector>> m_data;

    void copy(const ImageData &imageData);
    void move(ImageData &imageData);

    void fftShift2(ComplexVector *data);
    void fftShift3(ComplexVector *data);
};

#endif // IMAGEDATA_H
