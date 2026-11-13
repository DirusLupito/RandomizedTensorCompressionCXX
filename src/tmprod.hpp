#pragma once

#include "tensor.hpp"

#include <span>
#include <vector>

namespace tensor_compression {

/**
 * Computes the tensor-matrix product along the given mode, corresponding to
 * tmprod(T, M, mode) in MATLAB.
 *
 * For example, suppose T is an m x p x n tensor and M is an r x n matrix.
 * For each fixed pair of indices (i, j), Tmprod(T, M, 3) multiplies the column
 * vector formed by varying the third index by M. The result is an m x p x r
 * tensor. The other two dimensions are unchanged.
 *
 * @param tensor The tensor to multiply. Its entries must agree with its
 * dimensions. The input tensor is not modified.
 * @param matrix The matrix to apply along the selected mode. Its column count,
 * after applying the requested transpose, must match the size of that mode.
 * @param mode The mode to multiply along. Modes are numbered starting at 1,
 * as in MATLAB. A mode beyond the stored dimensions has size one.
 * @param transpose Whether to transpose the matrix before multiplying.
 * 'N' uses the matrix as provided, and is the default. 'T' transposes it,
 * while 'H' takes its conjugate transpose. Since our matrices are real,
 * 'T' and 'H' have the same effect.
 * @param savePermutation Whether to skip the final permutation which restores
 * the tensor's original axis order. Defaults to false.
 * @param inversePermutation An optional pointer to a vector which will receive
 * the permutation needed to restore the original axis order. Its entries are
 * numbered starting at 1, so they can be passed to MATLAB's permute function.
 * Providing this pointer also skips the final permutation, regardless of
 * savePermutation, matching MATLAB's two-output form. Defaults to nullptr.
 * The caller owns the vector, and the pointer is not retained.
 *
 * @return The tensor resulting from the multiplication. The selected dimension
 * becomes the matrix's row count after the requested transpose. If permutation
 * saving was requested, the selected axis is first and the other axes follow
 * in their original order.
 *
 * @throws std::invalid_argument if the tensor has fewer than two dimensions,
 * a negative dimension, or a number of entries that disagrees with its
 * dimensions, if the mode or transpose flag is invalid, or if the matrix's
 * dimensions are incompatible with the selected mode.
 * @throws std::length_error if the tensor dimensions or storage size cannot
 * be represented.
 * @throws std::bad_alloc if there is not enough memory for the result or
 * temporary data.
 */
Tensor Tmprod(const Tensor &tensor, const Eigen::MatrixXd &matrix,
              Eigen::Index mode, char transpose = 'N',
              bool savePermutation = false,
              std::vector<Eigen::Index> *inversePermutation = nullptr);

/**
 * Computes tensor-matrix products along several modes, corresponding to
 * tmprod(T, {U1, ...}, [mode1, ...]) in MATLAB. Each matrix is paired with
 * the mode at the same position in the modes list.
 *
 * The products are computed one at a time in the order used by Tensorlab.
 * Each selected axis is moved to the front so that the multiplication can be
 * expressed as an ordinary matrix product. The original axis order is restored
 * after all products are complete, unless the caller requests otherwise.
 * There is no explicit parallelism here. Each matrix product is handled by
 * Eigen.
 *
 * @param tensor The tensor to multiply. Its entries must agree with its
 * dimensions. The input tensor is not modified.
 * @param matrices The matrices to apply, one for each mode. Each matrix's
 * column count, after the requested transpose, must match the size of its
 * corresponding mode. The matrices are not modified, and the span is only
 * used during this call.
 * @param modes The modes to multiply along, numbered starting at 1. They must
 * be positive and distinct, with one mode per matrix. A mode beyond the stored
 * dimensions has size one. Empty matrices and modes lists return a copy of
 * the tensor.
 * @param transpose Whether to transpose the matrices before multiplying.
 * 'N' uses them as provided, and is the default. 'T' transposes every matrix.
 * 'H' takes every matrix's conjugate transpose, which has the same effect as
 * 'T' for our real matrices.
 * @param savePermutation Whether to skip the final permutation which restores
 * the tensor's original axis order. Defaults to false.
 * @param inversePermutation An optional pointer to a vector which will receive
 * the permutation needed to restore the original axis order. Its entries are
 * numbered starting at 1, as in MATLAB's permute function. Providing this
 * pointer also skips the final permutation, regardless of savePermutation.
 * Empty product lists give the identity permutation. Defaults to nullptr.
 * The caller owns the vector, and the pointer is not retained.
 *
 * @return The tensor resulting from all the multiplications. Each selected
 * dimension becomes its matrix's row count after the requested transpose.
 * The axes are returned in their original order unless permutation saving
 * was requested. Trailing dimensions of size one are retained.
 *
 * @throws std::invalid_argument if the tensor has fewer than two dimensions,
 * a negative dimension, or a number of entries that disagrees with its
 * dimensions, if the matrix and mode counts differ, if modes repeat or are
 * not positive, if a matrix has incompatible dimensions, or if the transpose
 * flag is invalid.
 * @throws std::length_error if the tensor dimensions or storage size cannot
 * be represented.
 * @throws std::bad_alloc if there is not enough memory for the result or
 * temporary data.
 */
Tensor Tmprod(const Tensor &tensor, std::span<const Eigen::MatrixXd> matrices,
              std::span<const Eigen::Index> modes, char transpose = 'N',
              bool savePermutation = false,
              std::vector<Eigen::Index> *inversePermutation = nullptr);

} // namespace tensor_compression
