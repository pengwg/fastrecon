#include <QString>
#include <QDir>
#include <QFile>
#include <QDebug>
#include <QApplication>
#include <QLabel>
#include <QElapsedTimer>

#include <float.h>

#include "ProgramOptions.h"
#include "ReconData.h"

#include "ConvKernel.h"
#include "GridLut.h"
#include "FFT2D.h"

#ifdef CUDA_CAPABLE
#include "FFTGpu.h"
#include "GridGpu.h"
#endif

void loadData(QVector<TrajPoint> &trajPoints, complexVector &trajData, ReconParameters &params)
{
    int trajSize = params.samples * params.projections;
    trajPoints.resize(trajSize);
    trajData.resize(trajSize);

    QFile file(params.traj_filename);
    file.open(QIODevice::ReadOnly);

    QVector<float> buffer(trajSize * 3);

    qint64 size = sizeof(float) * trajSize * 3;
    auto count = file.read((char *)buffer.data(), size);
    Q_ASSERT(count == size);

    file.close();

    float *pdata = buffer.data();
    for (int i = 0; i < trajSize; i++) {
        trajPoints[i].kx = pdata[0];
        trajPoints[i].ky = pdata[1];
        trajPoints[i].dcf = pdata[2];
        trajPoints[i].idx = i;
        pdata += 3;
    }

    file.setFileName(params.data_filename);
    file.open(QIODevice::ReadOnly);

    size = sizeof(float) * trajSize * 2;
    count = file.read((char *)buffer.data(), size);
    Q_ASSERT(count == size);

    file.close();

    pdata = buffer.data();
    for (int i = 0; i < trajSize; i++) {
        trajData[i] = std::complex<float> (pdata[0], pdata[1]);
        pdata += 2;
    }
}

template <class T>
void loadTraj(ReconData &reconData, const ReconParameters &params)
{
    int size = params.samples * params.projections;
    T *traj = new T(size);

    QFile file(params.traj_filename);
    file.open(QIODevice::ReadOnly);
    auto count = file.read((char *)traj->data(), size * sizeof(typename T::value_type));
    file.close();

    if (count != size * sizeof(typename T::value_type))
    {
        qWarning() << "Error: wrong data size in " << params.traj_filename << '\n';
        std::exit(1);
    }

    reconData.setTraj(traj);
}

void loadReconData(ReconData &reconData, const ReconParameters &params)
{
    // Load trajectory
    if (params.rczres > 1)
        loadTraj<Traj3D>(reconData, params);
    else
        loadTraj<Traj2D>(reconData, params);
            
    // Load data
    int size = params.samples * params.projections;
    KData *kdata = new KData(size);

    QFile file(params.data_filename);
    file.open(QIODevice::ReadOnly);
    auto count = file.read((char *)kdata->data(), size * sizeof(KData::value_type));
    file.close();

    if (count != size * sizeof(KData::value_type))
    {
        qWarning() << "Error: wrong data size in " << params.traj_filename << '\n';
        std::exit(1);
    }

    reconData.addChannelData(kdata);
}

void displayData(int n0, int n1, const complexVector& data, const QString& title)
{
    QVector<float> dataValue;

    float max = 0;
    float min = FLT_MAX;

    for (auto cValue : data) {
        float value = std::abs(cValue);
        if (value > max) max = value;
        if (value < min) min = value;

        dataValue << value;
    }

    QImage dataImage(n1, n0, QImage::Format_Indexed8);
    for (int i = 0; i < 256; i++) {
        dataImage.setColor(i, qRgb(i, i, i));
    }

    int i = 0;
    for (int y = 0; y < n0; y++) {
        auto imageLine = dataImage.scanLine(y);

        for (int x = 0; x < n1; x++) {
            uint idx;
            if (max == min)
                idx = 127;
            else
                idx = (dataValue[i] - min) / (max - min) * 255;
            imageLine[x] = idx;
            i++;
        }
    }

    QPixmap pixmap = QPixmap::fromImage(dataImage);

    QLabel *imgWnd = new QLabel("Image Window");
    imgWnd->setWindowTitle(title);
    imgWnd->setPixmap(pixmap);
    imgWnd->show();
}



int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    ProgramOptions options(argc, argv);
    options.showOptions();
    ReconParameters params = options.getReconParameters();

    ReconData reconData;
    loadReconData(reconData, params);


    QVector<TrajPoint> trajPoints;
    complexVector trajData;

    loadData(trajPoints, trajData, params);

    int kWidth = 4;
    float overGridFactor = params.overgridding_factor;
    ConvKernel kernel(kWidth, overGridFactor, 256);

    int gridSize = params.rcxres * overGridFactor;

    complexVector gDataCpu, gDataGpu;
    // KData gDataCpu
    QElapsedTimer timer;

    int rep = 100;
    qWarning() << "\nIteration" << rep << 'x';

    // CPU gridding
    GridLut gridCpu(gridSize, kernel);
    timer.start();
    for (int i = 0; i < rep; i++)
        gridCpu.gridding(trajPoints, trajData, gDataCpu);
    qWarning() << "\nCPU gridding time =" << timer.elapsed() << "ms";

#ifdef CUDA_CAPABLE
    // GPU gridding
    GridGpu gridGpu(gridSize, kernel);
    gridGpu.prepareGPU(trajPoints);

    timer.restart();
    for (int i = 0; i < rep; i++)
        gridGpu.transferData(trajData);

    cudaDeviceSynchronize();
    qWarning() << "\nGPU data transfer time =" << timer.elapsed() << "ms";

    timer.restart();
    for (int i = 0; i < rep; i++)
        gridGpu.gridding();

    cudaDeviceSynchronize();
    qWarning() << "\nGPU gridding time =" << timer.elapsed() << "ms";

    timer.restart();
    for (int i = 0; i < rep; i++)
        gridGpu.retrieveData(gDataGpu);
    qWarning() << "\nGPU data retrive time =" << timer.elapsed() << "ms";

    // CPU FFT
    FFT2D fft(gridSize, gridSize, false);
    timer.restart();
    for (int i = 0; i < rep; i++) {
        fft.fftShift(gDataCpu);
        fft.excute(gDataCpu);
        fft.fftShift(gDataCpu);
    }

    qWarning() << "\nCPU FFT time =" << timer.elapsed() << "ms";

    // GPU FFT
    FFTGpu fftGpu(gridSize, gridSize);
    timer.restart();
    for (int i = 0; i < rep; i++) {
        fftGpu.Execute((cufftComplex *)gridGpu.getDevicePointer());
    }
    cudaDeviceSynchronize();
    qWarning() << "\nGPU FFT time =" << timer.elapsed() << "ms";

#endif

    FFT2D fft(gridSize, gridSize, false);
    fft.fftShift(gDataCpu);
    fft.excute(gDataCpu);
    fft.fftShift(gDataCpu);

    if (options.isDisplay())
    {
        displayData(gridSize, gridSize, gDataCpu, "image");
        return app.exec();
    }
    else
        return 0;
}
