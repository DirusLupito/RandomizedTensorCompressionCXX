#pragma once

#include "tsvdm.hpp"

#include <Eigen/Core>

namespace tensor_compression {

/**
 * Rebuilds a tensor from its star-M SVD factors, computing U *M S *M V^T,
 * corresponding to reconstructFromTSVDM(U, S, V', M) in MATLAB. Applied to
 * truncated factors, this gives the compressed approximation of the original
 * tensor, so comparing the result with the original measures the error that
 * compression introduced.
 *
 * MATLAB computes the result with two star-M products, which each move their
 * operands to the transform domain, multiply slice by slice, and move back.
 * This does the same work in one pass: it moves all three factors to the
 * transform domain by multiplying along the third mode by M, computes
 * U(:, :, i) * S(:, :, i) * V(:, :, i)^T for each frontal slice, and returns
 * to the original domain by multiplying by M^T. The slices are processed one
 * at a time on the calling thread.
 *
 * @param factors The original domain factors to multiply, as returned by Tsvdm
 * with transformDomain false, in full, economy, or truncated form. U must be
 * m x r x n, S r x c x n, and V p x c x n. This takes V itself, as stored in
 * TsvdmFactors, where MATLAB's reconstructFromTSVDM takes its transpose. The
 * factors are not modified.
 * @param transform The n x n matrix M that the factors were computed with. It
 * must be orthogonal, since its transpose is used as its inverse.
 *
 * @return The m x p x n reconstructed tensor.
 *
 * @throws std::invalid_argument if a factor's dimensions or number of entries
 * are invalid, if the factor shapes do not fit together (see
 * ValidateTsvdmFactors), or if the transform is not n x n.
 * @throws std::length_error if a tensor's storage size cannot be represented.
 * @throws std::bad_alloc if there is not enough memory for the result or
 * temporary data.
 */
Tensor ReconstructFromTsvdm(const TsvdmFactors &factors,
                            const Eigen::MatrixXd &transform);

} // namespace tensor_compression
