#pragma once

#include "tensor.hpp"

namespace tensor_compression {

/**
 * Holds the three factors of a star-M tensor SVD, A = U *M S *M V^T. In the
 * transform domain, frontal slice i of the transformed tensor equals
 * U(:, :, i) * S(:, :, i) * V(:, :, i)^T, an ordinary matrix SVD.
 *
 * FIELDS:
 * - u: The m x r x n tensor of left singular vectors. In the transform domain,
 * each frontal slice has orthonormal columns.
 * - s: The r x c x n tensor of singular values. In the transform domain, each
 * frontal slice is diagonal, with nonnegative entries in descending order.
 * - v: The p x c x n tensor of right singular vectors. In the transform
 * domain, each frontal slice has orthonormal columns. This is V itself, not
 * V^T.
 *
 * Full factors have r = m and c = p, while economy factors have
 * r = c = min(m, p).
 */
struct TsvdmFactors {
    Tensor u;
    Tensor s;
    Tensor v;
};

/**
 * Computes the full star-M tensor SVD of a tensor with respect to an
 * orthogonal matrix, corresponding to TSVDMserial(A, M, transform, econ) in
 * MATLAB. This is the decomposition that both star-M SVD algorithms truncate.
 *
 * The tensor is first transformed along its third mode by M. Each frontal
 * slice of the transformed tensor is then decomposed with an ordinary matrix
 * SVD, independently of the others. Finally, the factors are returned to the
 * original domain by multiplying along the third mode by M^T, unless the
 * caller asks to keep them in the transform domain. The slices are currently
 * processed one at a time on the calling thread.
 *
 * @param tensor The m x p x n tensor to decompose. Its entries must agree with
 * its dimensions, and each dimension must be positive. A two-dimensional
 * tensor is treated as having n = 1. The tensor is not modified.
 * @param transform The n x n matrix M that defines the star-M product. It must
 * be orthogonal for the result to be a valid decomposition, which is the
 * caller's responsibility, since checking it would cost as much as a slice.
 * @param transformDomain Whether to return the factors in the transform domain
 * instead of the original domain. Defaults to false. Algorithm 2 uses the
 * transform domain factors directly.
 * @param economy Whether to compute economy-sized factors, as MATLAB's
 * svd(A, 'econ') does. Defaults to false. Economy factors are much smaller
 * when m and p differ greatly, and still contain everything needed to
 * reconstruct the tensor.
 *
 * @return The factors U, S, and V, in the original domain unless
 * transformDomain is true. See TsvdmFactors for their dimensions.
 *
 * @throws std::invalid_argument if the tensor's dimensions or number of
 * entries are invalid, if it has more than three dimensions or an empty
 * dimension, or if the transform is not n x n.
 * @throws std::runtime_error if Eigen fails to compute a slice's SVD.
 * @throws std::length_error if a tensor's storage size cannot be represented.
 * @throws std::bad_alloc if there is not enough memory for the factors or
 * temporary data.
 */
TsvdmFactors Tsvdm(const Tensor &tensor, const Eigen::MatrixXd &transform,
                   bool transformDomain = false, bool economy = false);

} // namespace tensor_compression
