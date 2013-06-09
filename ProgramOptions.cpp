#include <QDir>
#include <QSettings>
#include <argp.h>
#include <iostream>
#include <iomanip>

#include "ProgramOptions.h"

ProgramOptions::ProgramOptions(int argc, char *argv[])
{
    commandLineOptions(argc, argv);
    loadIniOptions();
}

error_t ProgramOptions::loadIniOptions()
{
    QFileInfo fileInfo(iniFileName);
    if (!fileInfo.exists())
    {
        qCritical("File '%s' does not exist!", qPrintable(iniFileName));
        exit(1);
    }

    QDir dir;
    QString path = dir.relativeFilePath(fileInfo.canonicalPath()) + '/';

    QSettings settings(iniFileName, QSettings::IniFormat);

    reconParameters.samples = settings.value("samples").toInt();
    reconParameters.projections = settings.value("projections").toInt();
    reconParameters.traj_filename = path + settings.value("traj_filename").toString();
    reconParameters.data_filename = path + settings.value("data_filename").toString();
    reconParameters.result_filename = path + settings.value("out_filename").toString();

    reconParameters.rcxres = settings.value("rcxres").toInt();
    reconParameters.rcyres = settings.value("rcyres").toInt();
    reconParameters.rczres = settings.value("rczres").toInt();

    if (reconParameters.overgridding_factor == 0)
        reconParameters.overgridding_factor = settings.value("overgridding_factor").toFloat();

    return 0;
}

error_t ProgramOptions::commandLineOptions(int argc, char *argv[])
{
    struct argp_option options[] =
    {
        { 0, 0, 0, 0, "Recon parameters:", 1},
        { 0, 'g', "FACTOR", 0, "Define over-gridding factor"},

        { 0, 0, 0, 0, "Miscellaneous:", 2},
        { "show", 777, 0, 0, "Display reconstruction in a window"},

        { 0, 0, 0, 0, "Help options:", -1},
        { 0, 'h', 0, OPTION_HIDDEN},
        {0}
    };

    struct argp argp =
    {
        options, parse_opt,
        "RECON.ini",
        "CPU & GPU MRI recon program.\v"
        "Command line options have precedence over RECON.ini."
    };

    parse_error = argp_parse(&argp, argc, argv, 0, 0, this);
    return parse_error;
}

error_t ProgramOptions::parse_opt(int key, char *arg, struct argp_state *state)
{
    ProgramOptions *parent = static_cast<ProgramOptions *> (state->input);

    switch (key)
    {
    case 777:
        parent->display = true;
        break;
    case 'g':
        parent->reconParameters.overgridding_factor = atof(arg);
        break;
    case 'h':
        argp_usage(state);
        break;
    case ARGP_KEY_ARG:
        parent->arg_count--;
        if (parent->arg_count >= 0)
            parent->iniFileName = QString(arg);
        break;
    case ARGP_KEY_END:
        if (parent->arg_count >= 1)
            argp_failure(state, 1, 0, "Missing RECON.ini arguments");
        else if (parent->arg_count < 0)
            argp_failure(state, 1, 0, "too many arguments");
        break;
    }
    return 0;
}

void ProgramOptions::showParameters() const
{
    std::cout << std::left
              << std::setw(22) << "Trajectory:" << reconParameters.traj_filename.toStdString() << std::endl
              << std::setw(22) << "Data:" << reconParameters.data_filename.toStdString() << std::endl
              << std::setw(22) << "Output:" << reconParameters.result_filename.toStdString()<< std::endl
              << std::setw(22) << "Samples:" << reconParameters.samples << std::endl
              << std::setw(22) << "Projections:" << reconParameters.projections << std::endl
              << std::setw(22) << "Overgridding factor:" << reconParameters.overgridding_factor << std::endl
              << std::setw(22) << "rcxres:" << reconParameters.rcxres << std::endl
              << std::setw(22) << "rcyres:" << reconParameters.rcyres << std::endl
              << std::setw(22) << "rczres:" << reconParameters.rczres << std::endl;
}

ReconParameters ProgramOptions::getReconParameters() const
{
    return reconParameters;
}
