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

ReconData loadReconData(const ReconParameters &params)
{
    ReconData reconData;
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
    return reconData;
}


int main(int argc, char *argv[])
{
    ProgramOptions options(argc, argv);
    options.showParameters();
    ReconParameters params = options.getReconParameters();

    // Load multi-channel data
    ReconData reconData = loadReconData(params);

    // Gridding kernel
    int kWidth = 3;
    float overGridFactor = params.overgridding_factor;
    ConvKernel kernel(kWidth, overGridFactor, 256);

    // CPU gridding
    int gridSize = params.rcxres * overGridFactor;
    GridLut gridCpu(gridSize, kernel);

    std::cout << "\nCPU gridding... " << std::endl;
    QElapsedTimer timer;
    timer.start();
    ImageData imgData = gridCpu.gridding(reconData);
    std::cout << "Gridding total time " << timer.elapsed() << " ms" << std::endl;


    // CPU FFT
    std::cout << "\nCPU FFT... " << std::endl;
    FFT fft(imgData.size());

    if (params.rczres > 1)
    {
        fft.plan(gridSize, gridSize, gridSize, false);
    }
    else
        fft.plan(gridSize, gridSize, false);

    timer.restart();

    // fft.fftShift(data);
    fft.excute(imgData);
    fft.fftShift(imgData);

    std::cout << "FFT total time " << timer.elapsed() << " ms" << std::endl;

    // Save result
    /*QFile file(params.result_filename);
    file.open(QIODevice::WriteOnly);
    auto count = file.write((const char *)data.data(), data.size() * sizeof(typename KData::value_type));
    file.close();*/

    // Display data
    int i = 0;
    int zSize = 0;
    if (params.rczres > 1) zSize = gridSize;

    if (options.isDisplay())
    {
        QApplication app(argc, argv);
        for (auto &data : imgData)
            displayData(*data.get(), gridSize, gridSize, zSize, QString("channel ") + QString::number(i++));
        return app.exec();
    }
    else
        return 0;
}
