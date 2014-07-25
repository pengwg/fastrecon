
template<typename T>
cuImageData<T>::cuImageData()
{
}

template<typename T>
cuImageData<T>::cuImageData(const int dim, const ImageSize &imageSize, LocalComplexVector *image)
    : ImageData<thrust::device_vector, T>(dim, imageSize, image)
{
}

template<typename T>
template<typename T1>
cuImageData<T>::cuImageData(T1 &&imageData)
{
    copy(std::forward<T1>(imageData));
}

template<typename T>
template<typename T1>
cuImageData<T> &cuImageData<T>::operator=(T1 &&imageData)
{
    copy(std::forward<T1>(imageData));
    return *this;
}

template<typename T>
void cuImageData<T>::copy(const basicImageData &imageData)
{
    const cuImageData &im = dynamic_cast<const cuImageData &>(imageData);
    m_dim = im.m_dim;
    m_size = im.m_size;
    m_data.clear();

    for (const auto &data : im.m_data)
    {
        auto data_copy = new LocalComplexVector(*data);
        addChannelImage(data_copy);
    }

    // std::cout << "Copy called" << std::endl;
}

template<typename T>
void cuImageData<T>::copy(basicImageData &&imageData)
{
    cuImageData &im = dynamic_cast<cuImageData &>(imageData);
    m_dim = im.m_dim;
    m_size = im.m_size;
    m_channels = im.m_channels;

    m_data = std::move(im.m_data);

    im.m_dim = 0;
    im.m_size = {0};
    im.m_channels = 0;
    // std::cout << "Move called" << std::endl;
}

template<typename T>
void cuImageData<T>::fftShift()
{
#pragma omp parallel for
    for (int n = 0; n < this->channels(); n++)
    {
        auto data = this->getChannelImage(n);

        if (this->m_dim == 3)
            fftShift3(data);
        else
            fftShift2(data);
    }
}

template<typename T>
void cuImageData<T>::fftShift2(LocalComplexVector  *data)
{

}

template<typename T>
void cuImageData<T>::fftShift3(LocalComplexVector *data)
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

//template class cuImageData<float>;
