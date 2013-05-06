#include "CommandLineOptions.h"
#include <argp.h>
#include <iostream>

CommandLineOptions::CommandLineOptions(int argc, char *argv[])
    : arg_count(3)
{
    struct argp_option options[] =
    {
        { "show", 's', 0, 0, "Display reconstruction in a window"},
        {0}
    };

    struct argp argp =
    {
        options, parse_opt,
        "traj_file data_file [output_file]",
        "\vOutput to recon.dat if out_file is not given"
    };

    parse_error = argp_parse (&argp, argc, argv, 0, 0, this);
    std::cout << "Trajectory: " << fileTraj.toStdString() << std::endl
              << "Data: " << fileData.toStdString() << std::endl
              << "Output: " << fileOut.toStdString() << std::endl;
}

error_t CommandLineOptions::parse_opt (int key, char *arg, struct argp_state *state)
{
    CommandLineOptions *parent = static_cast<CommandLineOptions *> (state->input);

    switch (key)
    {
    case 's':
        parent->display = true;
        break;

    case ARGP_KEY_ARG:
        switch (parent->arg_count)
        {
        case 3:
            parent->fileTraj = QString(arg);
            break;

        case 2:
            parent->fileData = QString(arg);
            break;

        case 1:
            parent->fileOut = QString(arg);
            break;
        }
        (parent->arg_count)--;
        break;

    case ARGP_KEY_END:
        if (parent->arg_count >= 2)
            argp_failure(state, 1, 0, "too few arguments");
        else if (parent->arg_count < 0)
            argp_failure (state, 1, 0, "too many arguments");
        break;
    }
    return 0;
}
