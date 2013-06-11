#include <QString>
#include <QDir>
#include <QFile>
#include <QApplication>
#include <QLabel>
#include <QElapsedTimer>

#include <iostream>
#include <float.h>

#include "ProgramOptions.h"
#include "ReconData.h"

#include "ConvKernel.h"
#include "GridLut.h"
#include "FFT.h"

#ifdef CUDA_CAPABLE
#include "FFTGpu.h"
#include "GridGpu.h"
#endif

void displayData(const KData& data, int n0, int n1, int n2, const QString& title)
{
    std::vector<float> dataValue;
    if (n2 == 0) n2 = 2;

    int start = (n0 * n1) * (n2 / 2 - 1);
    int end = (n0 * n1) * (n2 / 2);

    for (auto it = data.begin() + start; it < data.begin() + end; it++) {
        float value = std::abs(*it);
        dataValue.push_back(value);
    }

    float max = *std::max_element(dataValue.begin(), dataValue.end());
    float min = *std::min_element(dataValue.begin(), dataValue.end());

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

template <int N>
void loadReconData(ReconData<N> &reconData, const ReconParameters &params)
{
    // Load trajectory
    int size = params.samples * params.projections;
    Traj<N> *traj = new Traj<N>(size);

    QFile file(params.path + '/' + params.trajFiles);
    file.open(QIODevice::ReadOnly);
    auto count = file.read((char *)traj->data(), size * sizeof(typename Traj<N>::value_type));
    file.close();

    if (count != size * sizeof(typename Traj<N>::value_type))
    {
        std::cout << "Error: wrong data size in " << params.trajFiles.toStdString() << std::endl;
        std::exit(1);
    }

    reconData.setTraj(traj);

    // Load data
    size = params.samples * params.projections;
    KData *kdata = new KData(size);

    file.setFileName(params.path + '/' + params.dataFiles);
    file.open(QIODevice::ReadOnly);
    count = file.read((char *)kdata->data(), size * sizeof(KData::value_type));
    file.close();

    if (count != size * sizeof(KData::value_type))
    {
        std::cout << "Error: wrong data size in " << params.trajFiles.toStdString() << std::endl;
        std::exit(1);
    }

    reconData.addChannelData(kdata);
}

template <int N>
void gridding(const ReconParameters &params, KData &out)
{
    ReconData<N> reconData;
    loadReconData<>(reconData, params);

    int kWidth = 4;
    float overGridFactor = params.overgridding_factor;
    ConvKernel kernel(kWidth, overGridFactor, 256);

    int gridSize = params.rcxres * overGridFactor;

    int rep = 1;
    std::cout << "\nIteration " << rep << 'x' << std::endl;

    // CPU gridding
    GridLut gridCpu(gridSize, kernel);

    QElapsedTimer timer;
    timer.start();
    std::cout << "CPU gridding... " << std::flush;

    for (int i = 0; i < rep; i++)
        gridCpu.gridding(reconData, out);

    std::cout << timer.elapsed() << " ms";
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    ProgramOptions options(argc, argv);
    options.showParameters();
    ReconParameters params = options.getReconParameters();

    KData data;

    if (params.rczres > 1)
        gridding<3>(params, data);
    else
        gridding<2>(params, data);;

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

    int gridSize = params.rcxres * params.overgridding_factor;;
    int zSize = 0;

    FFT fft;
    if (params.rczres > 1)
    {
        fft.plan(gridSize, gridSize, gridSize, false);
        zSize = gridSize;
    }
    else
        fft.plan(gridSize, gridSize, false);

    QElapsedTimer timer;
    timer.start();

    std::cout << "  |  FFT... " << std::flush;
    // fft.fftShift(data);
    fft.excute(data);
    fft.fftShift(data);

    std::cout << timer.elapsed() << " ms" << std::endl;

    /*QFile file(params.result_filename);
    file.open(QIODevice::WriteOnly);
    auto count = file.write((const char *)data.data(), data.size() * sizeof(typename KData::value_type));
    file.close();*/

    if (options.isDisplay())
    {
        displayData(data, gridSize, gridSize, zSize, "image");
        return app.exec();
    }
    else
        return 0;
}
