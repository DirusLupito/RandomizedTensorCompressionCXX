#pragma once

#include "experiment.hpp"

#include <cstdint>

namespace tensor_compression::experiments {

/** Checks how the requested energy changes the ranks kept by star-M SVD II. */
class Tsvdm2EnergyExperiment final : public Experiment {
public:
    /**
     * @param threadCount The number of threads to use for the slice SVDs.
     * The serial experiment uses one and the parallel experiment uses two,
     * one for each slice of the test tensor.
     */
    explicit Tsvdm2EnergyExperiment(std::uint32_t threadCount = 1)
        : Experiment(threadCount == 1
                         ? "Serial star-M SVD II energy and truncation"
                         : "Parallel star-M SVD II energy and truncation"),
          concurrency(threadCount) {}

    /**
     * Sweeps gamma on a tensor with known singular values. Prints the rank
     * of each slice and checks the retained and discarded energies by
     * reconstructing the tensor at each level.
     *
     * @throws std::runtime_error if the ranks or reconstruction disagree
     * with the known spectrum, or if a slice decomposition fails.
     * @throws std::bad_alloc if there is not enough memory for the tensors.
     * @throws std::invalid_argument if the requested concurrency is zero.
     * @throws std::system_error if a worker thread cannot be started.
     */
    void RunExperiment() const override;

private:
    // The thread count used throughout the energy sweep.
    std::uint32_t concurrency;
};

} // namespace tensor_compression::experiments
