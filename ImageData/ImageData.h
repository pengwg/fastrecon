#ifndef IMAGEDATA_H
#define IMAGEDATA_H

#include <memory>
#include "common.h"

typedef struct
{
    int x;
    int y;
    int z;
} ImageSize;

template<typename T>
class ImageData
{
public:
    ImageData() {}
    ImageData(const int dim, const ImageSize &imageSize, std::unique_ptr<ComplexVector<T>> image = nullptr);

    ImageData(const ImageData<T> &imageData);
    ImageData(ImageData<T> &&imageData);

    // Copy
    ImageData<T> &operator=(const ImageData<T> &imageData);
    // Move
    ImageData<T> &operator=(ImageData<T> &&imageData);

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

    void addChannelImage(std::unique_ptr<ComplexVector<T>> image);

    const ComplexVector<T> *getChannelImage(int channel) const;

    ComplexVector<T> *getChannelImage(int channel);

    virtual void fftShift();
    virtual void lowFilter(int res);
    virtual void normalize();

protected:
    virtual void copy(const ImageData<T> &imageData);
    virtual void copy(ImageData<T> &&imageData);

    void fftShift2(ComplexVector<T> *data);
    void fftShift3(ComplexVector<T> *data);

    int m_dim = 0;
    ImageSize m_size = {0, 0, 0};
    int m_channels = 0;

    std::vector<std::unique_ptr<ComplexVector<T>>> m_data_multichannel;
};

#endif // IMAGEDATA_H
