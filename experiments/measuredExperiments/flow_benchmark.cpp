#include "flow_benchmark.hpp"

#include "tsvdm2.hpp"

#include <netcdf>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensor_compression::experiments {

FlowBenchmark::FlowBenchmark(std::string_view title, std::string_view fileStem,
                             const std::filesystem::path &dataDirectory,
                             bool measureBatches)
    : Experiment(title), stem(fileStem), directory(dataDirectory),
      batches(measureBatches) {}

/**
 * Reads horizontal velocity one frame at a time into MATLAB column-major order.
 * @param path The flow NetCDF file.
 * @param tangaroa Whether the source contains a depth axis.
 * @return For 2D flows T(y,t,x). For Tangaroa T(x + nx*y,t,z), with zero-based
 * indices.
 * @throws std::runtime_error if u has unexpected dimensions.
 * @throws netCDF::exceptions::NcException if the file or its data cannot be
 * read.
 * @throws std::length_error if the tensor size cannot be represented.
 * @throws std::bad_alloc if the frame or output does not fit in memory.
 */
static Tensor ReadVelocity(const std::filesystem::path &path, bool tangaroa) {
    const netCDF::NcFile file(path.string(), netCDF::NcFile::read);
    const auto u = file.getVar("u");
    if (u.isNull()) {
        throw std::runtime_error("Missing u in " + path.string());
    }
    const auto dims = u.getDims();
    const std::vector<std::string> axes =
        tangaroa ? std::vector<std::string>{"tdim", "zdim", "ydim", "xdim"}
                 : std::vector<std::string>{"tdim", "ydim", "xdim"};
    if (dims.size() != axes.size()) {
        throw std::runtime_error("Unexpected number of dimensions for u in " +
                                 path.string());
    }
    std::vector<std::size_t> count;
    for (std::size_t i = 0; i < axes.size(); ++i) {
        if (dims[i].getName() != axes[i] || dims[i].getSize() == 0) {
            throw std::runtime_error("Expected a nonempty " + axes[i] +
                                     " axis in " + path.string());
        }
        count.push_back(dims[i].getSize());
    }
    const Eigen::Index nx = count.back(), ny = count[count.size() - 2],
                       nt = count[0];
    const Eigen::Index nz = tangaroa ? count[1] : 1;
    const std::array<Eigen::Index, 2> plane{nx, ny};
    const Eigen::Index rows = tangaroa ? ElementCount(plane) : ny;
    const Eigen::Index slices = tangaroa ? nz : nx;
    Tensor tensor{{rows, nt, slices}, {}};
    tensor.values.resize(ElementCount(tensor.dimensions));
    Eigen::VectorXd frame(rows * slices);
    count[0] = 1;
    std::vector<std::size_t> start(count.size(), 0);
    for (Eigen::Index t = 0; t < nt; ++t) {
        start[0] = static_cast<std::size_t>(t);
        u.getVar(start, count, frame.data());
        for (Eigen::Index k = 0; k < slices; ++k) {
            auto column = tensor.values.segment(rows * (t + nt * k), rows);
            if (tangaroa) {
                // Each depth plane already has x varying fastest in the file.
                column = frame.segment(rows * k, rows);
            } else {
                // TeBaC transposes each x-by-y MATLAB frame before storing it.
                for (Eigen::Index y = 0; y < ny; ++y) {
                    column(y) = frame(y * nx + k);
                }
            }
        }
    }
    return tensor;
}

/**
 * @param n The number of frontal slices.
 * @return The orthonormal DCT-II matrix corresponding to dct(eye(n)).
 * @throws std::bad_alloc if the matrix cannot be allocated.
 */
static Eigen::MatrixXd DctMatrix(Eigen::Index n) {
    Eigen::MatrixXd matrix(n, n);
    for (Eigen::Index row = 0; row < n; ++row) {
        const double scale = std::sqrt((row == 0 ? 1.0 : 2.0) / n);
        for (Eigen::Index col = 0; col < n; ++col) {
            matrix(row, col) =
                scale * std::cos(std::numbers::pi * row * (col + 0.5) / n);
        }
    }
    return matrix;
}

void FlowBenchmark::RunExperiment() const {
    const Tensor tensor = ReadVelocity(directory / (std::string(stem) + ".nc"),
                                       stem == "tangaroa");
    const auto &dims = tensor.dimensions;
    const Eigen::Index maxRank = std::min(dims[0], dims[1]);
    if (batches && maxRank < 8) {
        throw std::runtime_error("The scheduler sweep requires slice rank 8.");
    }
    const Eigen::MatrixXd transform = DctMatrix(dims[2]);

    // The runner and manual menu use the same output directory. An incomplete
    // sweep remains available for inspection without replacing a completed one.
    const char *configured = std::getenv("RTC_MEASURED_OUTPUT");
    const std::filesystem::path output =
        configured && *configured ? configured : "build/measuredExperiments";
    std::filesystem::create_directories(output);
    const std::string sweep = batches ? "batch" : "scaling";
    const auto csv = output / (std::string(stem) + "_cpp_" + sweep + ".csv");
    const auto runId =
        std::chrono::system_clock::now().time_since_epoch().count();
    const std::filesystem::path partial =
        csv.string() + "." + std::to_string(runId) + ".partial";
    std::ofstream timings;
    timings.exceptions(std::ios::failbit | std::ios::badbit);
    timings.open(partial, std::ios::out | std::ios::noreplace);
    timings << "dataset,language,sweep,algorithm,execution,threads,batch_size,"
               "c,gamma,repeat,seconds,m,p,n\n";

    // Each driver starts before the clock. With one thread the caller runs
    // the serial loop. Every repetition computes all slice SVDs from scratch.
    for (const std::uint32_t threads : {1U, 2U, 4U, 8U, 16U, 32U}) {
        if (batches && threads != 32) {
            continue;
        }
        AsyncParallelForDriver driver(threads);
        for (const std::size_t batch : {1U, 2U, 4U, 8U, 16U, 32U}) {
            if (!batches && batch != 1) {
                break;
            }
            std::cout << stem << " (" << sweep << "), threads=" << threads
                      << ", batch=" << batch << " -> " << partial << std::endl;
            const auto record =
                [&](std::string_view algorithm, Eigen::Index c, double gamma,
                    int repetition,
                    std::chrono::steady_clock::duration duration) {
                    timings << stem << ",cpp," << sweep << ',' << algorithm
                            << ',' << (threads == 1 ? "serial" : "parallel")
                            << ',' << threads << ',' << batch << ',';
                    if (c) {
                        timings << c;
                    }
                    timings << ',';
                    if (gamma) {
                        timings << std::setprecision(6) << gamma;
                    }
                    timings << ',' << repetition << ',' << std::setprecision(17)
                            << std::chrono::duration<double>(duration).count()
                            << ',' << dims[0] << ',' << dims[1] << ','
                            << dims[2] << '\n'
                            << std::flush;
                };
            for (Eigen::Index c = batches ? 8 : 1; c <= (batches ? 8 : maxRank);
                 ++c) {
                const std::vector<Eigen::Index> ranks(dims[2], c);
                for (int repetition = 1; repetition <= 5; ++repetition) {
                    const auto start = std::chrono::steady_clock::now();
                    const auto factors = Tsvdm(tensor, transform, driver, ranks,
                                               true, true, batch);
                    record("TSVDM", c, 0, repetition,
                           std::chrono::steady_clock::now() - start);
                }
            }
            for (const double gamma : {0.9, 0.99, 0.999, 0.9999}) {
                if (batches && gamma != 0.99) {
                    continue;
                }
                for (int repetition = 1; repetition <= 5; ++repetition) {
                    const auto start = std::chrono::steady_clock::now();
                    const auto result = Tsvdm2(tensor, transform, driver, gamma,
                                               true, true, batch);
                    record("TSVDM2", 0, gamma, repetition,
                           std::chrono::steady_clock::now() - start);
                }
            }
        }
    }
    timings.close();
    std::filesystem::rename(partial, csv);
    std::cout << "Saved " << csv << '\n';
}

} // namespace tensor_compression::experiments
