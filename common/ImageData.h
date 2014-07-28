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
    ImageData(const ImageData &) = delete;
    ImageData &operator=(const ImageData &) = delete;

    void addChannelImage(LocalComplexVector *image);
    const LocalComplexVector *getChannelImage(int channel) const;
    LocalComplexVector *getChannelImage(int channel);

protected:
    ImageData() {}
    ImageData(const int dim, const ImageSize &imageSize, LocalComplexVector *image = nullptr);

    virtual void copy(const basicImageData &imageData) override;
    virtual void copy(basicImageData &&imageData) override;

    virtual ~ImageData() {}

    std::vector<std::unique_ptr<LocalComplexVector>> m_data;
};

#endif // IMAGEDATA_H
