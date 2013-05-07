#include <QDir>
#include <QSettings>
#include <argp.h>
#include <iostream>

#include "ProgramOptions.h"

ProgramOptions::ProgramOptions(int argc, char *argv[])
    : arg_count(3)
{
    QString path = QDir::currentPath();
    iniOptions(path);
    commandLineOptions(argc, argv);
}

error_t ProgramOptions::iniOptions(QString path)
{
    QSettings settings(path + "/input.ini", QSettings::IniFormat);

    reconParameters.samples = settings.value("samples").toInt();
    reconParameters.projections = settings.value("projections").toInt();
    reconParameters.overgridding_factor = settings.value("overgridding_factor").toFloat();
    reconParameters.traj_filename = settings.value("traj_filename").toString();
    reconParameters.data_filename = settings.value("data_filename").toString();
    reconParameters.out_filename = settings.value("out_filename").toString();

    return 0;
}

error_t ProgramOptions::commandLineOptions(int argc, char *argv[])
{
    struct argp_option options[] =
    {
        { 0, 's', 0, 0, "Display reconstruction in a window"},
        { 0, 't', "traj_filename", 0, ""},
        { 0, 'd', "data_filename", 0, ""},
        { 0, 'o', "out_filename", 0, ""},
        { 0, 'g', "og_factor", 0, ""},
        {0}
    };

    struct argp argp =
    {
        options, parse_opt,
        0,
        "Command line options override input.ini options."
    };

    parse_error = argp_parse (&argp, argc, argv, 0, 0, this);
    return parse_error;
}

error_t ProgramOptions::parse_opt (int key, char *arg, struct argp_state *state)
{
    ProgramOptions *parent = static_cast<ProgramOptions *> (state->input);

    switch (key)
    {
    case 's':
        parent->display = true;
        break;
    case 't':
        parent->reconParameters.traj_filename = QString(arg);
        break;
    case 'd':
        parent->reconParameters.data_filename = QString(arg);
        break;
    case 'o':
        parent->reconParameters.out_filename = QString(arg);
        break;
    case 'g':
        parent->reconParameters.overgridding_factor = atof(arg);
        break;
    }
    return 0;
}

void ProgramOptions::showOptions()
{
    std::cout << "Trajectory: " << reconParameters.traj_filename.toStdString() << std::endl
              << "Data: " << reconParameters.data_filename.toStdString() << std::endl
              << "Output: " << reconParameters.out_filename.toStdString()<< std::endl
              << "Samples: " << reconParameters.samples << std::endl
              << "Projections: " << reconParameters.projections << std::endl
              << "Overgridding factor: " << reconParameters.overgridding_factor << std::endl;
}

ReconParameters ProgramOptions::getReconParameters() const
{
    return reconParameters;
}
