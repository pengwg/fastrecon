#ifndef COMMANDLINEOPTIONS_H
#define COMMANDLINEOPTIONS_H

#include <QString>

struct ReconParameters
{
    QString path;
    QString trajFiles;
    QString dcfFile;
    QString dataFiles;
    QString outFile;

    int samples;
    int projections;
    float overgridding_factor = 0;
    float kernel_width = 4;
    bool pils = false;

    unsigned rcxres;
    unsigned rcyres;
    unsigned rczres;
};

class ProgramOptions
{
public:
    ProgramOptions(int argc, char *argv[]);
    void showParameters() const;

    bool isDisplay() const { return display; }
    bool isGPU() const { return gpu; }
    ReconParameters getReconParameters() const;

private:
    error_t loadIniOptions();
    error_t commandLineOptions(int argc, char *argv[]);

    static error_t parse_opt (int key, char *arg, struct argp_state *state);
    error_t parse_error;
    int arg_count = 1;

    bool display = false;
    bool gpu = false;
    ReconParameters reconParameters;
    QString iniFileName;
};

#endif // COMMANDLINEOPTIONS_H
