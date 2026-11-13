#pragma once

#include "tensor.hpp"

namespace tensor_compression {

/**
 * Reorders a tensor's axes, corresponding to MATLAB's permute function.
 * The entries are rearranged so the result is still in column-major order.
 *
 * @param tensor The tensor whose axes will be reordered. Its dimensions and
 * number of entries must agree. The input tensor is not modified.
 * @param axes The input axis to place at each position in the result,
 * numbered starting at 1 as in MATLAB. Every number from 1 through
 * axes.size() must appear exactly once, including every input axis.
 * Additional trailing axes have size one. For example, a two-dimensional
 * tensor can be permuted with {3, 1, 2}, treating its third dimension as one.
 *
 * @return A new tensor with its dimensions and entries in the requested order.
 * For example, axes {3, 1, 2} changes an m x p x n tensor to an n x m x p
 * tensor, with result(k, i, j) corresponding to tensor(i, j, k). The inverse
 * permutation returned by Tmprod can be passed directly as axes.
 *
 * @throws std::invalid_argument if the tensor's dimensions or number of
 * entries are invalid, if axes has fewer entries than the tensor has
 * dimensions, or if an axis is repeated or outside the range 1 through
 * axes.size().
 * @throws std::length_error if the tensor's storage size exceeds the range
 * of Eigen::Index.
 * @throws std::bad_alloc if there is not enough memory for the result or
 * temporary data.
 */
Tensor Permute(const Tensor &tensor, const std::vector<Eigen::Index> &axes);

} // namespace tensor_compression
