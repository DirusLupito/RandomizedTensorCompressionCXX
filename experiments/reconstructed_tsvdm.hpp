#pragma once

#include "experiment.hpp"

namespace tensor_compression::experiments {

/** Measures the error when reconstructing full and economy decompositions. */
class ReconstructedTsvdmExperiment final : public Experiment {
public:
    /** Sets the name displayed in the menu. */
    ReconstructedTsvdmExperiment() : Experiment("Star-M SVD reconstruction") {}

    /**
     * Checks that reconstructing untruncated star-M SVD factors gives back the
     * original tensor, on random tensors with a random orthogonal transform, in
     * both full and economy form. Economy factors drop only the parts of U and
     * V that multiply zeros in S, so they must reconstruct the tensor as well
     * as full factors do.
     *
     * @throws std::runtime_error if the reconstruction has the wrong dimensions
     * or differs from the original tensor by more than the tolerance.
     * @throws std::bad_alloc if there is not enough memory for the tensors or
     * temporary data.
     */
    void RunExperiment() const override;
};

} // namespace tensor_compression::experiments
