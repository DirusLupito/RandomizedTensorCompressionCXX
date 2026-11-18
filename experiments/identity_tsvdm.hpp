#pragma once

#include "experiment.hpp"

namespace tensor_compression::experiments {

/** Compares each identity-transform slice with its ordinary matrix SVD. */
class IdentityTsvdmExperiment final : public Experiment {
public:
    /** Sets the name displayed in the menu. */
    IdentityTsvdmExperiment()
        : Experiment("Star-M SVD with the identity transform") {}

    /**
     * Checks that the star-M SVD with the identity transform reduces to an
     * ordinary matrix SVD of each frontal slice. Multiplying by the identity
     * changes nothing, so every slice of U, S, and V must match Eigen's SVD of
     * the corresponding slice of the original tensor, in both full and economy
     * form. Both run the same algorithm on the same matrix, so the singular
     * vectors have the same signs and can be compared directly.
     *
     * @throws std::runtime_error if any factor slice differs from the matrix
     * SVD by more than the tolerance.
     * @throws std::bad_alloc if there is not enough memory for the tensors or
     * temporary data.
     */
    void RunExperiment() const override;
};

} // namespace tensor_compression::experiments
