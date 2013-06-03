#ifndef COMMANDLINEOPTIONS_H
#define COMMANDLINEOPTIONS_H

#include <QString>

struct ReconParameters
{
    QString traj_filename;
    QString data_filename;
    QString result_filename;

    int samples = 0;
    int projections = 0;
    float overgridding_factor = 0;
};

class ProgramOptions
{
public:
    ProgramOptions(int argc, char *argv[]);
    void showOptions();

    bool isDisplay() const { return display; }
    ReconParameters getReconParameters() const;

private:
    error_t iniOptions(QString path);
    error_t commandLineOptions(int argc, char *argv[]);

    static error_t parse_opt (int key, char *arg, struct argp_state *state);
    error_t parse_error;
    int arg_count = 1;

    bool display = false;
    ReconParameters reconParameters;
    QString iniFileName;
};

#endif // COMMANDLINEOPTIONS_H
