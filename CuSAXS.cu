#include "RunSaxs.h"
#include <iostream>
#include <vector>
#include <cstdlib> // for std::getenv
#include <CLI/CLI.hpp>
#include "Options.h"
#include "Ftypedefs.h"

/**
 * @brief Main entry point of the application.
 *
 * This function sets up and parses command-line arguments using the CLI11 library.
 * It extracts various options and parameters from the command line, such as the topology file,
 * trajectory file, grid size, start and end frames, and other optional parameters.
 * It then creates a RunSaxs object and calls its Run() method to execute the SAXS calculation.
 * Finally, it prints a completion message to the console.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @return 0 on successful completion, 1 on error.
 */
int main(int argc, char *argv[])
{
    CLI::App app{"App description"};

    std::string tpr_file;
    std::string xtc_file;
    std::vector<int> myN;
    std::vector<int> myNS;
    int start, end, dt{1};

    /**
     * @brief Adds various command-line options to the CLI11 app object.
     *
     * This function adds the following options to the CLI11 app object:
     * - Topology file (-s,--topology)
     * - Trajectory file (-x,--trajectory)
     * - Grid size (-g,--grid)
     * - Initial frame (-b,--begin)
     * - Last frame (-e,--end)
     * - Output file (-o,--out)
     * - Frame read interval (--dt)
     * - BSpline order (--order)
     * - Scaled grid size (--gridS)
     * - Grid scale factor (--Scale)
     * - Histogram bin size (--bin,--Dq)
     * - Reciprocal space cutoff (-q,--qcut)
     * - Water model (--water)
     * - Sodium atom count (--na)
     * - Chlorine atom count (--cl)
     * - Help flag (-h,--help)
     *
     * These options are used to configure the SAXS calculation performed by the
     * RunSaxs class.
     */
    // Add required option for topology file
    app.add_option("-s,--topology", tpr_file, "Topology file (.tpr)")->required();

    // Add optional option for trajectory file
    app.add_option("-x,--trajectory", xtc_file, "Trajectory file (.xtc)")->required();

    // Add optional option grid size
    app.add_option("-g,--grid ", myN, "Grid length")->required();

    // Add optional option grid size
    app.add_option("-b,--begin ", start, "Initial frame")->required();

    // Add optional option grid size
    app.add_option("-e,--end ", end, "Last Frame")->required();

    // Add optional cutoff on reciprocal space
    app.add_option("-o,--out", Options::outFile, "Model to use for the weighting function");

    // Add optional option grid size
    app.add_option("--dt ", dt, "Read two frames every dt");

    // Add optional option grid size
    app.add_option("--order ", Options::order, "BSpline order");

    // Add optional option grid size
    app.add_option("--gridS ", myNS, "Scaled Grid length");

    // Add optional option grid scale
    app.add_option("--Scale ", Options::sigma, "Multiplicative factor to obtained the largest scaled grid");

    // Add optional option grid scale
    app.add_option("--bin,--Dq ", Options::Dq, "Binsize of the histogram");

    // Add optional cutoff on reciprocal space
    app.add_option("-q,--qcut ", Options::Qcut, "Cutoff on reciprocal space");

    // Add optional cutoff on reciprocal space
    app.add_option("--water", Options::Wmodel, "Model to use for the weighting function");

    // Add optional cutoff on reciprocal space
    app.add_option("--na ", Options::Sodium, "Sodium atoms");

    // Add optional cutoff on reciprocal space
    app.add_option("--cl ", Options::Chlorine, "Chlorine atoms");

    // Add optional for constant pressure simulation
    app.add_option("--simulation, --simu", Options::Simulation, "Valid values are npt and nvt");

    // Add help option
    app.set_help_flag("-h,--help", "Print usage");

    CLI11_PARSE(app, argc, argv);
    int nx{0}, ny{0}, nz{0};
    if (myN.size() == 1)
    {
        nx = myN[XX];
        ny = myN[XX];
        nz = myN[XX];
    }
    else if (myN.size() == 3)
    {
        nx = myN[XX];
        ny = myN[YY];
        nz = myN[ZZ];
    }
    else
    {
        std::cout << "Invalid number of arguments for grid size. Please provide either 1 or 3 integers." << std::endl;
        return 1;
    }
    int nnx{0}, nny{0}, nnz{0};
    if (myNS.size() == 1)
    {
        nx = myNS[XX];
        ny = myNS[XX];
        nz = myNS[XX];
    }
    else if (myN.size() == 3)
    {
        nx = myNS[XX];
        ny = myNS[YY];
        nz = myNS[ZZ];
    }
    else if (myNS.size() != 0)
    {
        std::cout << "Invalid number of arguments for grid size. Please provide either 1 or 3 integers." << std::endl;
        return 1;
    }
    Options::tpr_file = tpr_file;
    Options::xtc_file = xtc_file;
    Options::nx = nx;
    Options::ny = ny;
    Options::nz = nz;
    Options::nnx = nnx;
    Options::nny = nny;
    Options::nnz = nnz;
    if (Options::Wmodel.size())
    {
        Options::myPadding = padding::given;
    }
    // Get the input string from the command-line arguments
    RunSaxs saxs(tpr_file, xtc_file);
    saxs.Run(start, end, dt);
    // run_python_script_with_args(tpr_file, xtc_file);

    // Print the result from the Python script
    std::cout << "CuSAXS calculation completed successfully" << std::endl;

    return 0;
}
