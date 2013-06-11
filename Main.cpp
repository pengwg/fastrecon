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

void displayData(const ComplexVector& data, int n0, int n1, int n2, const QString& title)
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

void loadReconData(ReconData &reconData, const ReconParameters &params)
{
    // Load trajectory
    int size = params.samples * params.projections;

    QDir dir(params.path, QString(params.trajFiles), QDir::Name);
    QStringList fileList = dir.entryList();

    std::cout << std::endl << "Read trajectory:" << std::endl;
    for (const QString &name : fileList)
    {
        FloatVector *traj = new FloatVector(size);

        QString fileName = params.path + '/' + name;
        std::cout << fileName.toStdString() << std::endl;

        QFile file(fileName);
        file.open(QIODevice::ReadOnly);
        auto count = file.read((char *)traj->data(), size * sizeof(FloatVector::value_type));
        file.close();

        if (count != size * sizeof(FloatVector::value_type))
        {
            std::cout << "Error: wrong data size in " << fileName.toStdString() << std::endl;
            std::exit(1);
        }

        reconData.addTrajComponent(traj);
    }

    // Load dcf
    std::cout << std::endl << "Read dcf:" << std::endl;
    FloatVector *dcf = new FloatVector(size);

    QString fileName = params.path + '/' + params.dcfFile;
    std::cout << fileName.toStdString() << std::endl;

    QFile file(fileName);
    file.open(QIODevice::ReadOnly);
    auto count = file.read((char *)dcf->data(), size * sizeof(FloatVector::value_type));
    file.close();

    if (count != size * sizeof(FloatVector::value_type))
    {
        std::cout << "Error: wrong data size in " << params.trajFiles.toStdString() << std::endl;
        std::exit(1);
    }
    reconData.setDcf(dcf);

    // Load data
    dir.setNameFilters(QStringList(params.dataFiles));
    fileList = dir.entryList();

    std::cout << std::endl << "Read data:" << std::endl;
    for (const QString &name : fileList)
    {
        ComplexVector *kdata = new ComplexVector(size);
        QString fileName = params.path + '/' + name;
        std::cout << fileName.toStdString() << std::endl;

        QFile file(fileName);
        file.open(QIODevice::ReadOnly);
        auto count = file.read((char *)kdata->data(), size * sizeof(ComplexVector::value_type));
        file.close();

        if (count != size * sizeof(ComplexVector::value_type))
        {
            std::cout << "Error: wrong data size in " << params.trajFiles.toStdString() << std::endl;
            std::exit(1);
        }

        reconData.addChannelData(kdata);
    }
}

void gridding(const ReconParameters &params, ComplexVector &out)
{
    ReconData reconData;
    loadReconData(reconData, params);

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

    ComplexVector data;
    gridding(params, data);

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
