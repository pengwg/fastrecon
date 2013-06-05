#ifndef COMMANDLINEOPTIONS_H
#define COMMANDLINEOPTIONS_H

#include <QString>

struct ReconParameters
{
    QString traj_filename;
    QString data_filename;
    QString result_filename;

    int samples;
    int projections;
    float overgridding_factor = 0;

    int rcxres;
    int rcyres;
    int rczres;
};

class ProgramOptions
{
public:
    ProgramOptions(int argc, char *argv[]);
    void showOptions() const;

    bool isDisplay() const { return display; }
    ReconParameters getReconParameters() const;

private:
    error_t loadIniOptions();
    error_t commandLineOptions(int argc, char *argv[]);

    static error_t parse_opt (int key, char *arg, struct argp_state *state);
    error_t parse_error;
    int arg_count = 1;

    bool display = false;
    ReconParameters reconParameters;
    QString iniFileName;
};

#endif // COMMANDLINEOPTIONS_H
