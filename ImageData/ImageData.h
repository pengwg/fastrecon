#ifndef IMAGEDATA_H
#define IMAGEDATA_H

#include <memory>
#include "common.h"

template<typename T>
class ImageData
{
public:
    ImageData() {}
    ImageData(const int dim, const ImageSize &imageSize, std::unique_ptr<ComplexVector<T>> image = nullptr);

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
    virtual void setChannels(int channels);

    ImageSize imageSize() const {
        return m_size;
    }

    int dim() const {
        return m_dim;
    }

    std::size_t dataSize() const;

    void addChannelImage(std::unique_ptr<ComplexVector<T>> image);
    const ComplexVector<T> *getChannelImage(int channel = 0) const;
    ComplexVector<T> *getChannelImage(int channel = 0);
    void updateChannelImage(ComplexVector<T> &&image, const ImageSize &size, int channel);

protected:
    virtual void copy(const ImageData<T> &imageData);
    virtual void move(ImageData<T> &imageData);

    int m_dim = 0;
    ImageSize m_size = {0, 0, 0};
    int m_channels = 0;

    std::vector<std::unique_ptr<ComplexVector<T>>> m_data_multichannel;
};

#endif // IMAGEDATA_H
