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

public:
    // Copy
    ImageData(const ImageData &imageData);
    // Move
    ImageData(ImageData &&imageData);
    // Copy
    ImageData &operator=(const ImageData &imageData);
    // Move
    ImageData &operator=(ImageData &&imageData);

    void addChannelImage(LocalComplexVector *image);
    const LocalComplexVector *getChannelImage(int channel) const;
    LocalComplexVector *getChannelImage(int channel);

protected:
    ImageData() {}
    ImageData(const int dim, const ImageSize &imageSize, LocalComplexVector *image = nullptr);

    virtual void copy(const basicImageData &imageData) override final;
    virtual void move(basicImageData &imageData) override final;
    virtual ~ImageData() {}

    std::vector<std::unique_ptr<LocalComplexVector>> m_data;
};

#endif // IMAGEDATA_H
