/**
 * @file ImprovedRunSaxs.cu
 * @brief Improved SAXS calculation implementation with optimized Python integration
 *
 * This file demonstrates how to integrate the new TrajectoryWrapper for better
 * performance and maintainability compared to the original RunSaxs implementation.
 */

#include "RunSaxs.h"
#include "TrajectoryWrapper.h"
#include "Options.h"
#include "saxsKernel.h"
#include "Cell.h"
#include "fmt/core.h"
#include "fmt/format.h"
#include <fstream>
#include <algorithm>
#include <chrono>

/**
 * @brief Improved RunSaxs implementation using the new TrajectoryWrapper
 *
 * Key improvements:
 * - Better error handling and resource management
 * - Batch frame processing for improved performance
 * - Comprehensive performance monitoring
 * - Cleaner separation of concerns
 */
class ImprovedRunSaxs
{
public:
    ImprovedRunSaxs(const std::string &tpr_file, const std::string &xtc_file)
        : tpr_file_(tpr_file), xtc_file_(xtc_file) {}

    /**
     * @brief Run SAXS calculation with improved Python integration
     *
     * @param start_frame First frame to process
     * @param end_frame Last frame to process
     * @param frame_step Frame interval
     */
    void run(int start_frame, int end_frame, int frame_step)
    {
        try
        {
            // Initialize trajectory wrapper
            trajectory_ = createTrajectoryWrapper(tpr_file_, xtc_file_, true);

            // Validate trajectory
            validateTrajectorySetup();

            // Initialize SAXS kernel
            initializeSaxsKernel();

            // Generate frame sequence
            std::vector<int> frame_sequence = generateFrameSequence(start_frame, end_frame, frame_step);

            fmt::print("Processing {} frames: {} to {} (step: {})\n",
                       frame_sequence.size(), start_frame, end_frame, frame_step);

            // Process frames in batches for better performance
            procesFramesBatched(frame_sequence);

            // Finalize and output results
            finalizeAndOutput();

            // Print performance statistics
            printPerformanceStats(frame_sequence.size());
        }
        catch (const std::exception &e)
        {
            fmt::print(stderr, "Error in SAXS calculation: {}\n", e.what());
            throw;
        }
    }

private:
    std::string tpr_file_;
    std::string xtc_file_;
    std::unique_ptr<TrajectoryWrapper> trajectory_;
    std::unique_ptr<saxsKernel> saxs_kernel_;
    std::chrono::high_resolution_clock::time_point start_time_;

    void validateTrajectorySetup()
    {
        auto validation_results = trajectory_->validateTrajectory();

        bool all_valid = true;
        for (const auto &[key, valid] : validation_results)
        {
            if (!valid)
            {
                fmt::print(stderr, "Validation failed for: {}\n", key);
                all_valid = false;
            }
        }

        if (!all_valid)
        {
            throw std::runtime_error("Trajectory validation failed");
        }

        // Get system information
        auto system_info = trajectory_->getSystemInfo();

        fmt::print("System information:\n");
        fmt::print("  Atoms: {}\n", system_info.n_atoms);
        fmt::print("  Frames: {}\n", system_info.n_frames);
        fmt::print("  Molecules: {}\n", system_info.n_molecules);
        fmt::print("  Proteins: {}, Waters: {}, Ions: {}\n",
                   system_info.n_proteins, system_info.n_waters, system_info.n_ions);
    }

    void initializeSaxsKernel()
    {
        start_time_ = std::chrono::high_resolution_clock::now();

        // Create SAXS kernel with current options
        saxs_kernel_ = std::make_unique<saxsKernel>(Options::nx, Options::ny, Options::nz, Options::order);
        saxs_kernel_->setnpx(8); // TODO: make this configurable
        saxs_kernel_->scaledCell();

        // Read first frame to get box dimensions
        auto first_frame = trajectory_->readFrame(0);
        Cell::calculateMatrices(first_frame.box_dimensions);
        auto oc = Cell::getOC();

        // Setup water model if specified
        setupWaterModel(first_frame.box_dimensions);

        // Initialize kernel memory and parameters
        saxs_kernel_->resetHistogramParameters(oc);
        saxs_kernel_->createMemory();
        saxs_kernel_->writeBanner();
        saxs_kernel_->setcufftPlan(Options::nnx, Options::nny, Options::nnz);

        fmt::print("SAXS kernel initialized with grid: {}x{}x{}\n",
                   Options::nx, Options::ny, Options::nz);
    }

    void setupWaterModel(const std::vector<std::vector<float>> &box_dimensions)
    {
        if (Options::myPadding != padding::given)
        {
            return;
        }

        auto atom_indices = trajectory_->getAtomIndices();

        // Auto-detect ion counts if not specified
        if (atom_indices.find("Na") != atom_indices.end() && Options::Sodium == 0)
        {
            Options::Sodium = static_cast<int>(atom_indices["Na"].size());
        }
        if (atom_indices.find("Cl") != atom_indices.end() && Options::Chlorine == 0)
        {
            Options::Chlorine = static_cast<int>(atom_indices["Cl"].size());
        }

        // Setup atom density calculator
        AtomCounter density_calc(
            box_dimensions[0][0], box_dimensions[1][1], box_dimensions[2][2],
            Options::Sodium, Options::Chlorine, Options::Wmodel,
            Options::nx, Options::ny, Options::nz);

        Options::myWmodel = density_calc.calculateAtomCounts();

        // Zero out contributions for missing atom types
        for (auto &[element, count] : Options::myWmodel)
        {
            if (atom_indices.find(element) == atom_indices.end())
            {
                count = 0.0f;
            }
        }

        fmt::print("Water model setup: Na={}, Cl={}\n", Options::Sodium, Options::Chlorine);
    }

    std::vector<int> generateFrameSequence(int start_frame, int end_frame, int step)
    {
        std::vector<int> sequence;

        int max_frames = trajectory_->getNumFrames();
        end_frame = std::min(end_frame, max_frames - 1);

        for (int frame = start_frame; frame <= end_frame; frame += step)
        {
            sequence.push_back(frame);
        }

        return sequence;
    }

    void procesFramesBatched(const std::vector<int> &frame_sequence)
    {
        const size_t batch_size = 10; // Process frames in batches of 10

        size_t total_frames = frame_sequence.size();

        for (size_t i = 0; i < total_frames; i += batch_size)
        {
            size_t end_idx = std::min(i + batch_size, total_frames);

            // Extract batch
            std::vector<int> batch(frame_sequence.begin() + i, frame_sequence.begin() + end_idx);

            // Process batch
            procesFrameBatch(batch);

            // Progress update
            fmt::print("Processed {}/{} frames\n", end_idx, total_frames);
        }
    }

    void procesFrameBatch(const std::vector<int> &frame_batch)
    {
        try
        {
            // Read batch of frames
            auto frames_data = trajectory_->readFramesBatch(frame_batch);

            // Process each frame
            for (size_t i = 0; i < frames_data.size(); ++i)
            {
                processFrame(frame_batch[i], frames_data[i]);
            }
        }
        catch (const std::exception &e)
        {
            fmt::print(stderr, "Error processing frame batch: {}\n", e.what());

            // Fallback to individual frame processing
            for (int frame_num : frame_batch)
            {
                try
                {
                    auto frame_data = trajectory_->readFrame(frame_num);
                    processFrame(frame_num, frame_data);
                }
                catch (const std::exception &frame_error)
                {
                    fmt::print(stderr, "Skipping frame {}: {}\n", frame_num, frame_error.what());
                }
            }
        }
    }

    void processFrame(int frame_number, const FrameData &frame_data)
    {
        try
        {
            // Convert coordinates from nm to Angström (factor of 10.0)
            std::vector<std::vector<float>> coords_angstrom = frame_data.coordinates;
            for (auto &atom_pos : coords_angstrom)
            {
                for (auto &coord : atom_pos)
                {
                    coord *= 10.0f;
                }
            }

            // Update cell matrices
            Cell::calculateMatrices(frame_data.box_dimensions);
            auto oc = Cell::getOC();

            // Get atom indices (cached in trajectory wrapper)
            auto atom_indices = trajectory_->getAtomIndices();

            // Run SAXS kernel
            saxs_kernel_->runPKernel(frame_number, frame_data.time, coords_angstrom, atom_indices, oc);
        }
        catch (const std::exception &e)
        {
            fmt::print(stderr, "Error processing frame {}: {}\n", frame_number, e.what());
            throw;
        }
    }

    void finalizeAndOutput()
    {
        // Get histogram if NVT simulation
        if (Options::Simulation == "nvt")
        {
            auto oc = Cell::getOC();
            saxs_kernel_->getHistogram(oc);
        }

        // Get SAXS profile
        auto saxs_profile = saxs_kernel_->getSaxs();

        // Write output file
        std::ofstream output_file(Options::outFile);
        if (!output_file)
        {
            throw std::runtime_error("Cannot open output file: " + Options::outFile);
        }

        for (const auto &data_point : saxs_profile)
        {
            output_file << std::fixed << std::setw(10) << std::setprecision(5) << data_point[0]
                        << std::scientific << std::setprecision(5) << std::setw(12) << data_point[1]
                        << std::endl;
        }

        output_file.close();

        fmt::print("Results written to: {}\n", Options::outFile);
    }

    void printPerformanceStats(size_t num_frames)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration<double>(end_time - start_time_).count();

        // Get performance stats from trajectory wrapper
        auto traj_stats = trajectory_->getPerformanceStats();
        auto cuda_time = saxs_kernel_->getCudaTime();

        double avg_total_time = total_duration / num_frames * 1000.0;    // ms per frame
        double avg_read_time = traj_stats["avg_read_time_cpp"] * 1000.0; // ms per frame

        std::string performance_banner = fmt::format(
            "\n=========================================================\n"
            "=                                                       =\n"
            "=                CuSAXS Performance Report           =\n"
            "=                                                       =\n"
            "=  Total Frames:       {:<6}                          =\n"
            "=  CUDA Time:          {:<10.2f} ms/frame             =\n"
            "=  Trajectory Read:    {:<10.2f} ms/frame             =\n"
            "=  Total Time:         {:<10.2f} ms/frame             =\n"
            "=  Overall Time:       {:<10.2f} seconds              =\n"
            "=                                                       =\n"
            "=  Memory Usage:       {:<10.1f} MB                   =\n"
            "=  Read Operations:    {:<6}                          =\n"
            "=                                                       =\n"
            "=========================================================\n",
            num_frames,
            cuda_time,
            avg_read_time,
            avg_total_time,
            total_duration,
            traj_stats.count("memory_usage_mb") ? traj_stats["memory_usage_mb"] : 0.0,
            static_cast<int>(traj_stats["read_count"]));

        fmt::print("{}", performance_banner);
    }
};

// Updated RunSaxs::Run method that uses the improved implementation
void RunSaxs::Run(int beg, int end, int dt)
{
    try
    {
        ImprovedRunSaxs improved_runner(tpr_file, xtc_file);
        improved_runner.run(beg, end, dt);

        fmt::print("CuSAXS calculation completed successfully\n");
    }
    catch (const std::exception &e)
    {
        fmt::print(stderr, "SAXS calculation failed: {}\n", e.what());
        throw;
    }
}

/**
 * @brief Demo function showing how to use the new trajectory interface directly
 */
void demonstrateNewInterface(const std::string &tpr_file, const std::string &xtc_file)
{
    try
    {
        // Create trajectory wrapper
        auto trajectory = createTrajectoryWrapper(tpr_file, xtc_file, true);

        // Get system information
        auto system_info = trajectory->getSystemInfo();
        fmt::print("System has {} atoms in {} frames\n",
                   system_info.n_atoms, system_info.n_frames);

        // Read a few frames
        std::vector<int> test_frames = {0, 10, 50, 100};
        auto frames_data = trajectory->readFrameBatch(test_frames);

        // Show frame information
        for (size_t i = 0; i < frames_data.size(); ++i)
        {
            const auto &frame = frames_data[i];
            fmt::print("Frame {}: time={:.2f} ps, {} atoms\n",
                       test_frames[i], frame.time, frame.coordinates.size());
        }

        // Show performance statistics
        auto perf_stats = trajectory->getPerformanceStats();
        fmt::print("Average read time: {:.4f} ms\n",
                   perf_stats["avg_read_time_cpp"] * 1000.0);
    }
    catch (const std::exception &e)
    {
        fmt::print(stderr, "Demo failed: {}\n", e.what());
    }
}