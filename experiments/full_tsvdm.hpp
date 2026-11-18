#pragma once

#include "experiment.hpp"

namespace tensor_compression::experiments {

/** Verifies the decomposition before and after returning to the original
 * domain. */
class FullTsvdmExperiment final : public Experiment {
public:
    /** Sets the name displayed in the menu. */
    FullTsvdmExperiment() : Experiment("Full and economy star-M SVD") {}

    /**
     * Checks the full star-M SVD on random tensors with a random orthogonal
     * transform, in both full and economy form. In the transform domain, each
     * frontal slice of U * S * V^T must reproduce the corresponding slice of
     * the transformed tensor, with nonnegative singular values in descending
     * order. The original domain factors must return to the transform domain
     * factors when multiplied by M, since M undoes the M^T that produced them.
     *
     * @throws std::runtime_error if a factor has the wrong dimensions, if the
     * slices are not reproduced, if the singular values are out of order, or if
     * the two domains disagree.
     * @throws std::bad_alloc if there is not enough memory for the tensors or
     * temporary data.
     */
    void RunExperiment() const override;
};

} // namespace tensor_compression::experiments
