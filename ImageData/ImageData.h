#ifndef IMAGEDATA_H
#define IMAGEDATA_H

#include <memory>
#include "common.h"

template<typename T>
class ImageData
{
public:
    ImageData() {}
    ImageData(const int dim, const ImageSize &imageSize);

    ImageData(const ImageData<T> &imageData);
    ImageData(ImageData<T> &&imageData);
    virtual ~ImageData() {}

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

    std::size_t dataSize() const;

    void addChannelImage(ComplexVector<T> &&image);
    const ComplexVector<T> *getChannelImage(int channel = 0) const;
    ComplexVector<T> *getChannelImage(int channel = 0);

protected:
    virtual void copy(const ImageData<T> &imageData);
    virtual void move(ImageData<T> &imageData);

    int m_dim = 0;
    ImageSize m_size = {0, 0, 0};
    int m_channels = 0;

    std::vector<ComplexVector<T>> m_data_multichannel;
};

#endif // IMAGEDATA_H
