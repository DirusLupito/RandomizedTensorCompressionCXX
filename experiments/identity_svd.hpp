#pragma once

#include "experiment.hpp"

namespace tensor_compression::experiments {

/** Checks Eigen against a matrix with three known singular values. */
class IdentitySvdExperiment final : public Experiment {
public:
    /** Sets the name displayed in the menu. */
    IdentitySvdExperiment() : Experiment("Eigen SVD of the identity matrix") {}

    /**
     * Prints the SVD of a 3x3 identity matrix and checks its reconstruction.
     *
     * @throws std::runtime_error if the decomposition fails either check.
     */
    void RunExperiment() const override;
};

} // namespace tensor_compression::experiments
