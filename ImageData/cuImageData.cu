#include <cassert>
#include <thrust/host_vector.h>
#include "cuImageData.h"

template<typename T>
cuImageData<T>::cuImageData()
{
}

template<typename T>
cuImageData<T>::cuImageData(const int dim, const ImageSize &imageSize)
    : ImageData<T>(dim, imageSize)
{
}

template<typename T>
cuImageData<T>::cuImageData(const cuImageData<T> &imageData)
{
    copy(imageData);
}

template<typename T>
cuImageData<T>::cuImageData(cuImageData<T> &&imageData)
{
    move(imageData);
}

// Copy
template<typename T>
cuImageData<T> &cuImageData<T>::operator=(const cuImageData<T> &imageData)
{
    copy(imageData);
    return *this;
}

// Move
template<typename T>
cuImageData<T> &cuImageData<T>::operator=(cuImageData<T> &&imageData)
{
    move(imageData);
    return *this;
}

template<typename T>
void cuImageData<T>::addChannelImage(std::unique_ptr<cuComplexVector<T>> image)
{
    if (image == nullptr) return;

    ComplexVector<T> out(image->size());
    auto im = reinterpret_cast<hostVector<typename cuComplexVector<T>::value_type> *>(&out);
    thrust::copy(image->begin(), image->end(), im->begin());

    m_cu_data = std::move(image);

    //auto h_im = reinterpret_cast<ComplexVector<T> *>(im);
    //thrust::copy(image->begin(), image->end(), h_im->begin());
    //auto image_ptr = thrust::raw_pointer_cast(image->data());
    //cudaMemcpy(im->data(), image_ptr, im->size() * sizeof(typename ComplexVector<T>::value_type), cudaMemcpyDeviceToHost);
    //thrust::host_vector<typename cuComplexVector<T>::value_type> h_im(*image);

    ImageData<T>::addChannelImage(std::move(out));
    m_channel_in_device = this->m_channels - 1;
}

template<typename T>
cuComplexVector<T> *cuImageData<T>::cuGetChannelImage(int channel)
{
    if (channel == m_channel_in_device)
    {
        return m_cu_data.get();
    }
    else if (channel < this->channels())
    {
        auto &h_data = reinterpret_cast<hostVector<typename cuComplexVector<T>::value_type> &>
                (this->m_data_multichannel[channel]);
        if (m_cu_data == nullptr)
            m_cu_data.reset(new cuComplexVector<T>(h_data));
        else
            *m_cu_data = h_data;

        m_channel_in_device = channel;
        return m_cu_data.get();
    }
    else
        return nullptr;
}

template<typename T>
void cuImageData<T>::setChannels(int channels)
{
    ImageData<T>::setChannels(channels);
    if (m_channel_in_device > channels - 1)
        invalidateDevice();
}

template<typename T>
void cuImageData<T>::syncDeviceToHost()
{
    if (m_channel_in_device == -1) return;

    auto &h_data = reinterpret_cast<hostVector<typename cuComplexVector<T>::value_type> &>
            (this->m_data_multichannel[m_channel_in_device]);
    thrust::copy(m_cu_data->begin(), m_cu_data->end(), h_data.begin());
}

template<typename T>
void cuImageData<T>::invalidateDevice()
{
    m_channel_in_device = -1;
}

template<typename T>
void cuImageData<T>::copy(const ImageData<T> &imageData)
{
    ImageData<T>::copy(imageData);
    m_channel_in_device = -1;

    auto im = dynamic_cast<const cuImageData<T> *>(&imageData);
    if (im)
    {
        m_channel_in_device = im->m_channel_in_device;
        if (m_cu_data == nullptr)
            m_cu_data.reset(new cuComplexVector<T>(*im->m_cu_data));
        else
            *m_cu_data = *im->m_cu_data;

        std::cout << "-- cuImageData: copy --" << std::endl;
    }
}

template<typename T>
void cuImageData<T>::move(ImageData<T> &imageData)
{
    ImageData<T>::move(imageData);
    m_channel_in_device = -1;

    auto im = dynamic_cast<cuImageData<T> *>(&imageData);
    if (im)
    {
        m_channel_in_device = im->m_channel_in_device;
        if (m_cu_data == nullptr)
            m_cu_data.reset(new cuComplexVector<T>);

        m_cu_data->swap(*im->m_cu_data);
        im->m_channel_in_device = -1;
        std::cout << "-- cuImageData: move --" << std::endl;
    }
}

template class cuImageData<float>;
