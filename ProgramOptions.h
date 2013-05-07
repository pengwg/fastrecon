#ifndef COMMANDLINEOPTIONS_H
#define COMMANDLINEOPTIONS_H

#include <QString>

struct ReconParameters
{
    QString traj_filename;
    QString data_filename;
    QString out_filename;

    int samples;
    int projections;
    float overgridding_factor;
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
    int arg_count;

    bool display = false;
    ReconParameters reconParameters;
};

#endif // COMMANDLINEOPTIONS_H
