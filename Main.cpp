#include <QString>
#include <QDir>
#include <QFile>
#include <QApplication>
#include <QLabel>
#include <QElapsedTimer>

#include <iostream>
#include <float.h>
#include <omp.h>

#include "ProgramOptions.h"
#include "hostReconData.h"
#include "cuReconData.h"
#include "hostImageData.h"
#include "cuImageData.h"
#include "ImageRecon.h"
#include "ConvKernel.h"
#include "GridLut.h"
#include "FFT.h"

#ifdef CUDA_CAPABLE
#include "FFTGpu.h"
#include "GridGpu.h"
#endif

void displayData(const ComplexVector& data, ImageSize size, const QString& title)
{
    std::vector<float> dataValue;
    int n0 = size.x;
    int n1 = size.y;
    int n2 = size.z;

    if (n2 < 2) n2 = 2;

    int nImages = 4;

    int start = (n0 * n1) * (n2 / 2 - 3);
    int end = (n0 * n1) * (n2 / 2 - 3 + nImages);

    for (auto it = data.begin() + start; it < data.begin() + end; it++) {
        float value = std::abs(*it);
        dataValue.push_back(value);
    }

    float max = *std::max_element(dataValue.begin(), dataValue.end());
    float min = *std::min_element(dataValue.begin(), dataValue.end());

    QImage dataImage(n1 * nImages, n0, QImage::Format_Indexed8);
    for (int i = 0; i < 256; i++) {
        dataImage.setColor(i, qRgb(i, i, i));
    }

    int i = 0;
    for (int y = 0; y < n0; y++) {
        auto imageLine = dataImage.scanLine(y);
        i = y * n1;
        for (int j = 0; j < nImages; j++)
        {
            for (int x = j * n0; x < j * n0 + n1; x++)
            {
                Q_ASSERT(i < dataValue.size());
                uint idx;
                if (max == min)
                    idx = 127;
                else
                    idx = (dataValue[i] - min) / (max - min) * 255;
                imageLine[x] = idx;
                i++;
            }
            i += n0 * (n1 - 1);
        }
    }

    QPixmap pixmap = QPixmap::fromImage(dataImage);

    QLabel *imgWnd = new QLabel("Image Window");
    imgWnd->setWindowTitle(title);
    imgWnd->setPixmap(pixmap);
    imgWnd->show();
}

void loadReconData(const ReconParameters &params, basicReconData<float> *reconData)
{
    QDir dir(params.path, QString(params.trajFiles), QDir::Name);
    QStringList trajFileList = dir.entryList();
    for (QString &name : trajFileList)
    {
        name = params.path + name;
    }

    dir.setNameFilters(QStringList(params.dataFiles));
    QStringList dataFileList = dir.entryList();
    for (QString &name : dataFileList)
    {
        name = params.path + name;
    }

    QString dcfFileName = params.path + params.dcfFile;

    reconData->loadFromFiles(dataFileList, trajFileList, dcfFileName);
}


int main(int argc, char *argv[])
{
    ProgramOptions options(argc, argv);
    options.showParameters();
    ReconParameters params = options.getReconParameters();

    // -------------- Load multi-channel data -----------------
    int size = params.samples * params.projections;
    auto reconData = new hostReconData<float>(size);
    loadReconData(params, reconData);

    // GPU Testing
    auto d_reconData = new cuReconData<float>(size);
    loadReconData(params, d_reconData);

    omp_set_num_threads(std::min(reconData->channels(), omp_get_num_procs()));

    QElapsedTimer timer0, timer;
    timer0.start();

    // -------------- Gridding kernel ------------------------
    int kWidth = params.kernel_width;
    float overGridFactor = params.overgridding_factor;
    ConvKernel kernel(kWidth, overGridFactor, 512);

    // -------------- Gridding -------------------------------
    std::cout << "\nCPU gridding... " << std::endl;
    int gridSize = params.rcxres * overGridFactor;
    GridLut gridCpu(gridSize, kernel);

    timer.start();
    hostImageData<float> imgData = gridCpu.gridding(*reconData);
    //CUDA testing
    cuImageData<float> cuimgData = gridCpu.gridding(*d_reconData);

    std::cout << "Gridding total time " << timer.elapsed() << " ms" << std::endl;

    hostImageData<float> imgMap;
    if (params.pils)
        imgMap = imgData;

    // --------------- FFT ----------------------------------
    std::cout << "\nCPU FFT... " << std::endl;
    FFT fft(reconData->rcDim(), {gridSize, gridSize, gridSize});

    timer.restart();
    // fft.fftShift(data);
    fft.excute(imgData);
    imgData.fftShift();
    std::cout << "FFT total time " << timer.restart() << " ms" << std::endl;

    // -------------- Recon Methods -----------------------------------
    ImageRecon recon(imgData, {params.rcxres, params.rcyres, params.rczres});
    hostImageData<float> finalImage;

    timer.restart();
    if (params.pils) {
        std::cout << "\nRecon PILS... " << std::endl;
        imgMap.lowFilter(22);
        std::cout << "\nLow pass filtering | " << timer.restart() << " ms" << std::endl;

        std::cout << "\nFFT low res image... " << std::endl;
        fft.excute(imgMap);
        imgMap.fftShift();
        std::cout << "FFT total time " << timer.restart() << " ms" << std::endl;

        imgMap.normalize();

        std::cout << "\nSum of Square Field Map..." << std::flush;
        finalImage = recon.SOS(imgMap);
        std::cout << " | " << timer.elapsed() << " ms" << std::endl;
    }
    else
    {
        std::cout << "\nRecon SOS... " << std::flush;
        finalImage = recon.SOS();
        std::cout << " | " << timer.elapsed() << " ms" << std::endl;
    }

    std::cout << "\nProgram total time excluding I/O: " << timer0.elapsed() / 1000.0 << " s" << std::endl;

    delete reconData;
    delete d_reconData;
    // -------------------------- Save Data ---------------------------
    /*QFile file(params.result_filename);
    file.open(QIODevice::WriteOnly);
    auto count = file.write((const char *)data.data(), data.size() * sizeof(typename KData::value_type));
    file.close();*/

    // -------------------------- Display Data -----------------------
    int n = 0;
    if (options.isDisplay())
    {
        QApplication app(argc, argv);
        for (int i = 0; i < finalImage.channels(); i++)
        {
            auto data = finalImage.getChannelImage(i);
            displayData(*data, finalImage.imageSize(), QString("channel ") + QString::number(n++));
        }
        return app.exec();
    }
    else
        return 0;
}
