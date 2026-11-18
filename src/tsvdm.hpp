#pragma once

#include "async_parallel_for_driver.hpp"
#include "tensor.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <span>

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
 * r = c = min(m, p). Truncated factors have r = c = the largest rank kept in
 * any slice.
 */
struct TsvdmFactors {
    Tensor u;
    Tensor s;
    Tensor v;
};

/**
 * Checks that star-M SVD factors are valid tensors whose shapes fit together,
 * so that U(:, :, i) * S(:, :, i) * V(:, :, i)^T is defined for every slice.
 * Functions that take factors call this before reading their entries. MATLAB
 * reports a mismatch as an indexing or assignment error partway through, but
 * this check reports it before any work is done.
 *
 * @param factors The factors to check. They are not modified.
 *
 * @throws std::invalid_argument if a factor's dimensions or number of entries
 * are invalid, if a factor does not have exactly three dimensions, if U's
 * column count differs from S's row count, if V's column count differs from
 * S's column count, or if the factors have different numbers of slices.
 * @throws std::length_error if a factor's storage size cannot be represented.
 */
void ValidateTsvdmFactors(const TsvdmFactors &factors);

/**
 * Computes the star-M tensor SVD of a tensor with respect to an orthogonal
 * matrix, optionally truncated, corresponding to
 * TSVDM(A, M, rho, transform, econ) in MATLAB. This is the decomposition
 * that both star-M SVD algorithms truncate.
 *
 * The tensor is first transformed along its third mode by M. Each frontal
 * slice of the transformed tensor is then decomposed with a full matrix SVD,
 * independently of the others. If ranks are given, each slice is then
 * truncated to its rank with TruncateTsvdm, as MATLAB does, rather than
 * computing a partial SVD. Finally, the factors are returned to the original
 * domain by multiplying along the third mode by M^T, unless the caller asks
 * to keep them in the transform domain.
 *
 * The slice SVDs run in parallel on the given driver, like MATLAB's parfor.
 * Threads claim batches of consecutive slices from a shared counter until
 * none remain, so faster threads take more slices. A driver with a
 * concurrency of 1 runs every slice on the calling thread, which corresponds
 * to TSVDMserial. Every slice is computed the same way whichever thread runs
 * it, so the factors are identical for any concurrency and batch size.
 *
 * @param tensor The m x p x n tensor to decompose. Its entries must agree with
 * its dimensions, and each dimension must be positive. A two-dimensional
 * tensor is treated as having n = 1. The tensor is not modified.
 * @param transform The n x n matrix M that defines the star-M product. It must
 * be orthogonal for the result to be a valid decomposition, which is the
 * caller's responsibility, since checking it would cost as much as a slice.
 * @param driver The driver whose threads compute the slice SVDs. The caller
 * creates it once and reuses it, so starting threads is not part of the
 * decomposition's running time, just as MATLAB's parpool is started before
 * TSVDM is called. It must not be running another ParallelFor from a
 * different thread.
 * @param ranks The number of singular triples to keep in each frontal slice,
 * in slice order, corresponding to MATLAB's rho. Defaults to empty, which
 * keeps every triple. Otherwise there must be one rank per slice, each between
 * 0 and min(m, p). See TruncateTsvdm. As in MATLAB, the ranks are checked only
 * once the slice SVDs are done.
 * @param transformDomain Whether to return the factors in the transform domain
 * instead of the original domain. Defaults to false. Algorithm 2 uses the
 * transform domain factors directly.
 * @param economy Whether to compute economy-sized factors, as MATLAB's
 * svd(A, 'econ') does. Defaults to false. Economy factors are much smaller
 * when m and p differ greatly, and still contain everything needed to
 * reconstruct the tensor.
 * @param batchSize The number of consecutive slices a thread claims at once.
 * Must be positive. Defaults to 1, which balances uneven slice costs best. A
 * batch size of ceil(n / concurrency) gives each thread at most one
 * contiguous block, which is close to a static split.
 *
 * @return The factors U, S, and V, in the original domain unless
 * transformDomain is true. See TsvdmFactors for their dimensions.
 *
 * @throws std::invalid_argument if the tensor's dimensions or number of
 * entries are invalid, if it has more than three dimensions or an empty
 * dimension, if the transform is not n x n, or if ranks are given but there
 * is not exactly one per slice or one is outside [0, min(m, p)], or if
 * batchSize is zero.
 * @throws std::runtime_error if Eigen fails to compute a slice's SVD. When
 * several slices fail, the first failure is reported, after every thread has
 * stopped.
 * @throws std::length_error if a tensor's storage size cannot be represented.
 * @throws std::bad_alloc if there is not enough memory for the factors or
 * temporary data.
 */
TsvdmFactors Tsvdm(const Tensor &tensor, const Eigen::MatrixXd &transform,
                   AsyncParallelForDriver &driver,
                   std::span<const Eigen::Index> ranks = {},
                   bool transformDomain = false, bool economy = false,
                   std::size_t batchSize = 1);

} // namespace tensor_compression
