#pragma once

#include "tmprod.hpp"

namespace tensor_compression {

/**
 * Computes the mode-3 tensor-matrix product, corresponding to
 * mode3TensorMatrixProduct(T, M) in MATLAB. This simply calls Tmprod with
 * mode 3, so both functions use the same multiplication and axis ordering.
 *
 * @param tensor The tensor to multiply. For an m x p x n tensor, the first two
 * dimensions remain unchanged. A missing third dimension is treated as size
 * one, as in MATLAB.
 * @param matrix The matrix to apply along the third mode. For a tensor whose
 * third dimension has size n, the matrix must have n columns.
 *
 * @return The resulting tensor in its original axis order. An m x p x n
 * tensor multiplied by an r x n matrix gives an m x p x r tensor.
 *
 * @throws std::invalid_argument if the tensor's dimensions or number of
 * entries are invalid, or if the matrix's column count does not match the
 * third dimension.
 * @throws std::length_error if the tensor dimensions or storage size cannot
 * be represented.
 * @throws std::bad_alloc if there is not enough memory for the result or
 * temporary data.
 */
inline Tensor Mode3TensorMatrixProduct(const Tensor &tensor,
                                       const Eigen::MatrixXd &matrix) {
    return Tmprod(tensor, matrix, 3);
}

} // namespace tensor_compression
