#ifndef IMAGEFILTER_H
#define IMAGEFILTER_H

#include <fftw3.h>
#include "ImageData.h"

class FFT;

template<typename T>
class ImageFilter
{
public:
    static std::shared_ptr<ImageFilter<T>> Create(ImageData<T> &imageData);
    virtual ~ImageFilter();

    void setNumOfThreads(unsigned threads) {
        m_num_threads = threads;
    }
    void fftShift();
    virtual void lowFilter(int res);
    virtual void normalize();
    virtual void crop(const ImageSize &imageSize);
    virtual void SOS(ImageSize reconSize);
    virtual void SOS(const ImageData<T> &map, ImageSize reconSize);
    virtual void fftPlan(int sign = FFTW_BACKWARD);
    void fftExecute();

protected:
    ImageFilter(ImageData<T> &imageData);
    virtual void fftShift2(ComplexVector<T> *data);
    virtual void fftShift3(ComplexVector<T> *data);

    FFT *m_fft = nullptr;

private:
    ImageData<T> &m_associatedData;
    unsigned m_num_threads = 1;
};

#endif // IMAGEFILTER_H
