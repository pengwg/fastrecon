#ifndef IMAGEDATA_H
#define IMAGEDATA_H

#include <memory>
#include "basicImageData.h"
#include "common.h"

template<template<typename, typename> class C, typename T>
class ImageData : public basicImageData
{
public:
    typedef typename LocalVectorType<C, T>::type LocalVector;
    typedef typename LocalComplexVectorType<C, T>::type LocalComplexVector;

    ImageData();
    ImageData(const int dim, const ImageSize &imageSize, LocalComplexVector *image = nullptr);
    virtual ~ImageData() {}

    // Copy
    ImageData(const ImageData &imageData);
    // Move
    ImageData(ImageData &&imageData);
    // Copy
    ImageData &operator=(const ImageData &ImageData);
    // Move
    ImageData &operator=(ImageData &&ImageData);

    void addChannelImage(LocalComplexVector *image);
    const LocalComplexVector *getChannelImage(int channel) const;
    LocalComplexVector *getChannelImage(int channel);

    virtual void fftShift() override;
    virtual void lowFilter(int res) override;
    virtual void normalize() override;

private:
    std::vector<std::unique_ptr<LocalComplexVector>> m_data;

    virtual void copy(const basicImageData &imageData) override;
    virtual void move(basicImageData &imageData) override;

    void fftShift2(hostComplexVector<T> *data);
    void fftShift3(hostComplexVector<T> *data);
    void fftShift2(cuComplexVector<T> *data) {}
    void fftShift3(cuComplexVector<T> *data) {}
};

#endif // IMAGEDATA_H
