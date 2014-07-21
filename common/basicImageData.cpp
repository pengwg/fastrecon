#include <iostream>
#include <math.h>

#include "basicImageData.h"

basicImageData::basicImageData()
{
}

basicImageData::basicImageData(int dim, const ImageSize &size)
    : m_dim(dim), m_size(size)
{

}

int basicImageData::dataSize() const
{
    if (m_dim == 3)
        return m_size.x * m_size.y * m_size.z;
    else
        return m_size.x * m_size.y;
}
