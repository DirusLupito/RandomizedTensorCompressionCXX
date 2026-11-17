#pragma once

#include "tsvdm.hpp"

#include <Eigen/Core>

#include <span>

namespace tensor_compression {

/**
 * Truncates star-M SVD factors to a separate rank for each frontal slice,
 * corresponding to truncateTSVDM(U, S, V, rho) in MATLAB. This is the
 * compression step of both star-M SVD algorithms. Algorithm 1 uses the same
 * rank for every slice, while Algorithm 2 chooses each slice's rank from a
 * global energy threshold.
 *
 * Slice i keeps its first ranks[i] singular triples: the leading columns of U
 * and V, and the leading square block of S. Since the singular values are in
 * descending order, these are the largest ones, so in the transform domain each
 * truncated slice is the best approximation of the transformed slice with that
 * rank. Every slice is padded with zeros to the largest rank, as in MATLAB, so
 * that the result is still a tensor.
 *
 * @param factors The factors to truncate, normally transform domain factors from
 * Tsvdm, in full or economy form. Truncating original domain factors does not
 * give a meaningful approximation, because the slices only decouple in the
 * transform domain. Each factor must have exactly three dimensions, with U
 * m x q x n, S q x c x n, and V p x c x n. The singular values on the diagonal
 * of each S slice must be in descending order. The factors are not modified.
 * @param ranks The number of singular triples to keep in each frontal slice, in
 * slice order. There must be one rank per slice, each between 0 and min(q, c),
 * which is min(m, p) for factors from Tsvdm. A uniform rank k is n copies of k.
 * A rank of 0 removes the slice's contribution entirely.
 *
 * @return The truncated factors, with U m x r x n, S r x r x n, and V p x r x n,
 * where r is the largest rank, or 0 if there are no slices. In each slice, the
 * columns beyond the slice's own rank are zero.
 *
 * @throws std::invalid_argument if a factor's dimensions or number of entries
 * are invalid, if a factor does not have exactly three dimensions, if the
 * factor shapes do not fit together, if there is not exactly one rank per
 * slice, or if any rank is negative or greater than min(q, c).
 * @throws std::length_error if a factor's storage size cannot be represented.
 * @throws std::bad_alloc if there is not enough memory for the truncated
 * factors.
 */
TsvdmFactors TruncateTsvdm(const TsvdmFactors &factors,
                           std::span<const Eigen::Index> ranks);

} // namespace tensor_compression
