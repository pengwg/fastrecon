#ifndef COMMANDLINEOPTIONS_H
#define COMMANDLINEOPTIONS_H

#include <QString>

class CommandLineOptions
{
public:
    CommandLineOptions(int argc, char *argv[]);

    QString getTrajFileName() { return fileTraj; }
    QString getDataFileName() { return fileData; }
    QString getOutputName() { return fileOut; }
    bool isDisplay() { return display; }

private:
    static error_t parse_opt (int key, char *arg, struct argp_state *state);
    error_t parse_error;
    int arg_count;

    bool display = false;
    QString fileTraj;
    QString fileData;
    QString fileOut = QString("recon.dat");
};

#endif // COMMANDLINEOPTIONS_H
