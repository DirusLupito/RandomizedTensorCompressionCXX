#pragma once

#include "experiment.hpp"

namespace tensor_compression::experiments {

/** Checks the shapes and entries of factors truncated to per-slice ranks. */
class TruncatedTsvdmExperiment final : public Experiment {
public:
    /** Sets the name displayed in the menu. */
    TruncatedTsvdmExperiment() : Experiment("Star-M SVD truncation") {}

    /**
     * Checks truncation of the star-M SVD on random tensors with a random
     * orthogonal transform, in both full and economy form. Each truncated slice
     * must hold exactly the leading singular triples of the untruncated
     * transform domain factors, padded with zeros to the largest rank.
     * Truncating inside Tsvdm must give exactly the same factors as calling
     * TruncateTsvdm on its untruncated output. Ranks that do not fit the tensor
     * must be rejected.
     *
     * @throws std::runtime_error if truncated factors have the wrong dimensions
     * or entries, if Tsvdm's truncation disagrees with TruncateTsvdm, or if
     * invalid ranks are accepted.
     * @throws std::bad_alloc if there is not enough memory for the tensors or
     * temporary data.
     */
    void RunExperiment() const override;
};

} // namespace tensor_compression::experiments
