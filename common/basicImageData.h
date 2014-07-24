#ifndef BASICIMAGEDATA_H
#define BASICIMAGEDATA_H

typedef struct
{
    int x;
    int y;
    int z;
} ImageSize;

class basicImageData
{
public:
    // Copy
    basicImageData &operator=(const basicImageData &ImageData);
    // Move
    basicImageData &operator=(basicImageData &&ImageData);

    int channels() const {
        return m_channels;
    }

    ImageSize imageSize() const {
        return m_size;
    }

    int dim() const {
        return m_dim;
    }

    int dataSize() const;

    virtual void fftShift() = 0;
    virtual void lowFilter(int res) = 0;
    virtual void normalize() = 0;

protected:
    basicImageData();
    basicImageData(int dim, const ImageSize &size);
    virtual ~basicImageData() {}

    virtual void copy(const basicImageData &imageData) = 0;
    virtual void move(basicImageData &imageData) = 0;

    int m_dim = 0;
    ImageSize m_size = {0, 0, 0};
    int m_channels = 0;
};

#endif // BASICIMAGEDATA_H
