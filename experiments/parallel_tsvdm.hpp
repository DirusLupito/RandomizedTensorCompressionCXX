#pragma once

#include "experiment.hpp"

namespace tensor_compression::experiments {

/** Compares the outputs for several thread counts and batch sizes. */
class ParallelTsvdmExperiment final : public Experiment {
public:
    /** Sets the name displayed in the menu. */
    ParallelTsvdmExperiment() : Experiment("Serial and parallel star-M SVD") {}

    /**
     * Checks that the parallel star-M SVD gives exactly the same factors as the
     * serial one, for several thread counts and batch sizes. Each slice's SVD
     * is computed the same way whichever thread runs it, so any difference at
     * all, even in the last bit, means that threads interfered with each
     * other's slices or that a slice was skipped or computed twice. Also checks
     * that a slice SVD failure on a worker thread reaches the caller as an
     * exception.
     *
     * The thread counts include one that does not divide the number of slices
     * and one larger than the number of slices. The batch sizes cover fully
     * dynamic claiming (1), a small batch (2), and one contiguous block per
     * thread, which is close to a static split. A 200-slice tensor makes the
     * threads compete for slices many times.
     *
     * @throws std::runtime_error if any parallel factors differ from the serial
     * factors, or if a slice SVD failure is not reported.
     * @throws std::system_error if a worker thread cannot be started.
     * @throws std::bad_alloc if there is not enough memory for the tensors or
     * temporary data.
     */
    void RunExperiment() const override;
};

} // namespace tensor_compression::experiments
