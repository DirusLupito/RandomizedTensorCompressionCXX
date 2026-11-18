#include "truncate_tsvdm.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tensor_compression {

TsvdmFactors TruncateTsvdm(const TsvdmFactors &factors,
                           std::span<const Eigen::Index> ranks) {
    ValidateTsvdmFactors(factors);

    // Get the tensor dimensions. MATLAB reads m and p from size(S), which only
    // agrees with U and V for full factors. Economy S slices are
    // min(m, p) x min(m, p), so reading U and V directly keeps truncation
    // working for both forms.
    const Eigen::Index m = factors.u.dimensions[0];
    const Eigen::Index p = factors.v.dimensions[0];
    const Eigen::Index n = factors.s.dimensions[2];

    // A slice cannot keep more singular triples than it has. MATLAB reports
    // these cases as out-of-range indexing errors inside the loop below.
    const Eigen::Index maxRank =
        std::min(factors.s.dimensions[0], factors.s.dimensions[1]);
    if (static_cast<Eigen::Index>(ranks.size()) != n) {
        throw std::invalid_argument(
            "Truncation needs exactly one rank per frontal slice.");
    }
    for (const Eigen::Index rank : ranks) {
        if (rank < 0 || rank > maxRank) {
            throw std::invalid_argument(
                "Each truncation rank must be between 0 and the number of "
                "singular values in a slice.");
        }
    }

    // Initialize the truncated tensors. Every slice is padded to the largest
    // rank so the result is still a tensor, and the entries start at zero
    // because slices with smaller ranks only fill their leading columns.
    const Eigen::Index maxRho = ranks.empty() ? 0 : std::ranges::max(ranks);
    TsvdmFactors truncated;
    truncated.u.dimensions = {m, maxRho, n};
    truncated.u.values.setZero(ElementCount(truncated.u.dimensions));
    truncated.s.dimensions = {maxRho, maxRho, n};
    truncated.s.values.setZero(ElementCount(truncated.s.dimensions));
    truncated.v.dimensions = {p, maxRho, n};
    truncated.v.values.setZero(ElementCount(truncated.v.dimensions));

    // Keep the leading singular triples of each slice, so that slice i of the
    // truncated A hat is U(:, 1:rho_i, i) * S(1:rho_i, 1:rho_i, i) *
    // V(:, 1:rho_i, i)^T.
    for (Eigen::Index i = 0; i < n; ++i) {
        const Eigen::Index rank = ranks[static_cast<std::size_t>(i)];
        FrontalSlice(truncated.u, i).leftCols(rank) =
            FrontalSlice(factors.u, i).leftCols(rank);
        FrontalSlice(truncated.s, i).topLeftCorner(rank, rank) =
            FrontalSlice(factors.s, i).topLeftCorner(rank, rank);
        FrontalSlice(truncated.v, i).leftCols(rank) =
            FrontalSlice(factors.v, i).leftCols(rank);
    }
    return truncated;
}

} // namespace tensor_compression
