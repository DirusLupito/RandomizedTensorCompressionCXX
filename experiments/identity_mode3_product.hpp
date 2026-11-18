#pragma once

#include "experiment.hpp"

namespace tensor_compression::experiments {

/** Checks mode-3 multiplication with a transform that preserves every entry. */
class IdentityMode3ProductExperiment final : public Experiment {
public:
    /** Sets the name displayed in the menu. */
    IdentityMode3ProductExperiment()
        : Experiment("Mode-3 product with the identity matrix") {}

    /**
     * Checks that multiplying a random tensor along mode 3 by the identity
     * matrix leaves the tensor unchanged. Prints the Frobenius norms and each
     * pair of input and output entries so the result can also be inspected
     * directly.
     *
     * @throws std::runtime_error if the product changes the tensor's
     * dimensions, entries, or Frobenius norm, or if the norm of the difference
     * is nonzero.
     * @throws std::bad_alloc if there is not enough memory for the tensors or
     * temporary data.
     */
    void RunExperiment() const override;
};

} // namespace tensor_compression::experiments
