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

    // Copy
    ImageData(const basicImageData &imageData);
    ImageData(const ImageData &imageData);
    // Move
    ImageData(basicImageData &&imageData);
    ImageData(ImageData &&imageData);

    // Copy
    ImageData &operator=(const basicImageData &ImageData);
    ImageData &operator=(const ImageData &ImageData);
    // Move
    ImageData &operator=(basicImageData &&ImageData);
    ImageData &operator=(ImageData &&ImageData);

    void addChannelImage(LocalComplexVector *image);
    const LocalComplexVector *getChannelImage(int channel) const;
    LocalComplexVector *getChannelImage(int channel);

    virtual void fftShift() override;
    virtual void lowFilter(int res) override;
    virtual void normalize() override;

private:
    std::vector<std::unique_ptr<LocalComplexVector>> m_data;

    void copy(const basicImageData &imageData);
    void move(basicImageData &imageData);

    void fftShift2(LocalComplexVector *data);
    void fftShift3(LocalComplexVector *data);
};

#endif // IMAGEDATA_H
