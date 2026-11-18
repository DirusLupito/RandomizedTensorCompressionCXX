#include "parallel_tsvdm.hpp"

#include "experiment_helpers.hpp"
#include "tsvdm.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tensor_compression::experiments {

void ParallelTsvdmExperiment::RunExperiment() const {
    const std::array<std::array<Eigen::Index, 3>, 3> shapes{
        {{6, 4, 5}, {3, 5, 4}, {8, 6, 200}}};
    std::mt19937 generator(548);

    // The serial reference runs every slice on the calling thread.
    AsyncParallelForDriver serial(1);
    std::vector<std::unique_ptr<AsyncParallelForDriver>> parallel;
    for (const std::uint32_t concurrency : {2U, 3U, 8U}) {
        parallel.push_back(
            std::make_unique<AsyncParallelForDriver>(concurrency));
    }

    for (const auto &[m, p, n] : shapes) {
        const Tensor tensor = RandomTensor({m, p, n}, generator);
        const Eigen::MatrixXd transform = RandomOrthogonal(n, generator);
        const std::vector<Eigen::Index> noRanks;
        const std::vector<Eigen::Index> uniformRanks(
            static_cast<std::size_t>(n), 2);
        std::size_t runs = 0;
        for (const bool economy : {false, true}) {
            for (const std::vector<Eigen::Index> *ranks :
                 {&noRanks, &uniformRanks}) {
                for (const bool transformDomain : {true, false}) {
                    const TsvdmFactors expected =
                        Tsvdm(tensor, transform, serial, *ranks,
                              transformDomain, economy);
                    for (const auto &driver : parallel) {
                        const std::size_t count = static_cast<std::size_t>(n);
                        const std::size_t threads = driver->GetConcurrency();
                        const std::size_t block =
                            (count + threads - 1) / threads;
                        for (const std::size_t batchSize :
                             {std::size_t{1}, std::size_t{2}, block}) {
                            const TsvdmFactors actual =
                                Tsvdm(tensor, transform, *driver, *ranks,
                                      transformDomain, economy, batchSize);
                            if (!SameTensor(actual.u, expected.u) ||
                                !SameTensor(actual.s, expected.s) ||
                                !SameTensor(actual.v, expected.v)) {
                                throw std::runtime_error(
                                    "Parallel star-M SVD differs from the "
                                    "serial result.");
                            }
                            ++runs;
                        }
                    }
                }
            }
        }
        std::cout << "Parallel star-M SVD of a random " << m << "x" << p << "x"
                  << n << " tensor: " << runs
                  << " runs on 2, 3, and 8 threads match the serial result "
                     "bit for bit\n";
    }

    // A NaN makes Eigen report invalid input for the slice SVDs. The mode-3
    // product spreads it to every slice, so the worker threads fail too, and
    // the first failure must be rethrown on the calling thread.
    Tensor invalid = RandomTensor({8, 6, 200}, generator);
    invalid.values(0) = std::numeric_limits<double>::quiet_NaN();
    bool failureReported = false;
    try {
        Tsvdm(invalid, RandomOrthogonal(200, generator), *parallel.back());
    } catch (const std::runtime_error &) {
        failureReported = true;
    }
    if (!failureReported) {
        throw std::runtime_error(
            "A failed slice SVD on a worker thread was not reported.");
    }
    std::cout << "Parallel star-M SVD verification passed.\n\n";
}

} // namespace tensor_compression::experiments
