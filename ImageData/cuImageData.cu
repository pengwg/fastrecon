#include <cassert>
#include <thrust/host_vector.h>
#include "cuImageData.h"

template<typename T>
cuImageData<T>::cuImageData()
{
}

template<typename T>
cuImageData<T>::cuImageData(const int dim, const ImageSize &imageSize, cuComplexVector<T> *image)
    : ImageData<T>(dim, imageSize)
{
    addChannelImage(image);
}

template<typename T>
void cuImageData<T>::addChannelImage(cuComplexVector<T> *image)
{
    if (image == nullptr) return;

    m_cu_data.reset(image);
    auto im = new hostVector<typename cuComplexVector<T>::value_type>(*image);

    //auto h_im = reinterpret_cast<ComplexVector<T> *>(im);
    //thrust::copy(image->begin(), image->end(), h_im->begin());
    //auto image_ptr = thrust::raw_pointer_cast(image->data());
    //cudaMemcpy(im->data(), image_ptr, im->size() * sizeof(typename ComplexVector<T>::value_type), cudaMemcpyDeviceToHost);
    //thrust::host_vector<typename cuComplexVector<T>::value_type> h_im(*image);

    addChannelImage(reinterpret_cast<ComplexVector<T> *>(im));
    m_channel_in_device = this->m_channels;
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
        auto h_data = reinterpret_cast<std::vector<typename cuComplexVector<T>::value_type> &>
                (*this->m_data_multichannel[channel]);
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
void cuImageData<T>::copy(const ImageData<T> &imageData)
{
    std::cout << "cuImageData: copy ";

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

        std::cout << " copy(device)" << std::endl;
    }
}

template<typename T>
void cuImageData<T>::copy(ImageData<T> &&imageData)
{
    std::cout << "cuImageData: move ";

    ImageData<T>::copy(std::move(imageData));
    m_channel_in_device = -1;

    auto im = dynamic_cast<cuImageData<T> *>(&imageData);
    if (im)
    {
        m_channel_in_device = im->m_channel_in_device;
        if (m_cu_data == nullptr)
            m_cu_data.reset(new cuComplexVector<T>);

        m_cu_data->swap(*im->m_cu_data);
        im->m_channel_in_device = -1;
        std::cout << " move(device)" << std::endl;
    }
}

template<typename T>
void cuImageData<T>::fftShift()
{
    for (int n = 0; n < this->channels(); n++)
    {
        auto data = cuGetChannelImage(n);

        if (this->m_dim == 3)
            fftShift3(data);
        else
            fftShift2(data);
    }
}

template<typename T>
void cuImageData<T>::fftShift2(cuComplexVector<T>  *data)
{

}

template<typename T>
void cuImageData<T>::fftShift3(cuComplexVector<T> *data)
{

}

template<typename T>
void cuImageData<T>::lowFilter(int res)
{

}

template<typename T>
void cuImageData<T>::normalize()
{

}

template class cuImageData<float>;
