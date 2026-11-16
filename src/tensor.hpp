#pragma once

#include <Eigen/Core>

#include <span>
#include <vector>

namespace tensor_compression {

/**
 * Represents a tensor whose entries are real double precision numbers.
 * The dimensions are stored separately from the entries, which are kept in
 * one vector in MATLAB's column-major order.
 *
 * For example, in an m x p x n tensor, the first index changes fastest,
 * followed by the second, and then the third. Using zero-based indices,
 * the entry T(i, j, k) is stored at i + m * (j + p * k) in values.
 *
 * FIELDS:
 * - dimensions: The size of the tensor along each axis. There must be at least
 * two dimensions, as in MATLAB, and each size must be nonnegative. A size of
 * zero means the tensor has no entries. Trailing dimensions of size one are
 * retained.
 * - values: The vector containing the tensor's entries. Its length must equal
 * the product of the dimensions. The tensor owns this memory through Eigen.
 */
struct Tensor {
    std::vector<Eigen::Index> dimensions;
    Eigen::VectorXd values;
};

/**
 * Computes the number of entries required for a tensor with the given
 * dimensions, checking that its storage size can be represented.
 *
 * @param dimensions The size of the tensor along each axis. Each size must
 * be nonnegative.
 *
 * @return The product of the dimensions, or zero if any dimension is zero.
 * An empty list of dimensions gives a product of one.
 *
 * @throws std::invalid_argument if any dimension is negative.
 * @throws std::length_error if the number of bytes needed for the entries
 * exceeds the range of Eigen::Index.
 */
Eigen::Index ElementCount(std::span<const Eigen::Index> dimensions);

/**
 * Checks that a tensor's dimensions describe the entries it actually stores.
 * Tensor operations call this before using the dimensions to calculate
 * positions in the data vector. The tensor is not modified.
 *
 * @param tensor The tensor to check. It must have at least two nonnegative
 * dimensions, and its number of entries must equal their product.
 *
 * @throws std::invalid_argument if the tensor has fewer than two dimensions,
 * a negative dimension, or a number of entries that disagrees with its
 * dimensions.
 * @throws std::length_error if the storage size exceeds the range of
 * Eigen::Index.
 */
void ValidateTensor(const Tensor &tensor);

/**
 * Views one frontal slice of a tensor as a matrix, corresponding to
 * T(:, :, slice + 1) in MATLAB. Because the entries are stored in column-major
 * order, each frontal slice occupies one contiguous block of the data vector,
 * so the view refers to the tensor's own entries rather than a copy. Writing
 * through the view modifies the tensor.
 *
 * @param tensor The tensor to view. It must have two or three dimensions, and
 * its entries must agree with them. A two-dimensional tensor has a single
 * frontal slice, as in MATLAB. The tensor must outlive the returned view, and
 * its entries must not be reallocated while the view is in use.
 * @param slice The zero-based index of the frontal slice, from 0 up to but not
 * including the size of the third dimension.
 *
 * @return An m x p matrix view of the slice, where m and p are the first two
 * dimensions of the tensor.
 *
 * @throws std::invalid_argument if the tensor's dimensions or number of
 * entries are invalid, or if it has more than three dimensions.
 * @throws std::out_of_range if the slice index is outside the third dimension.
 * @throws std::length_error if the storage size exceeds the range of
 * Eigen::Index.
 */
Eigen::Map<Eigen::MatrixXd> FrontalSlice(Tensor &tensor, Eigen::Index slice);

/**
 * Views one frontal slice of a tensor as a read-only matrix. This behaves
 * exactly like the modifiable overload, but the entries cannot be changed
 * through the view.
 *
 * @param tensor The tensor to view. It must have two or three dimensions, and
 * its entries must agree with them. The tensor must outlive the returned view.
 * @param slice The zero-based index of the frontal slice, from 0 up to but not
 * including the size of the third dimension.
 *
 * @return An m x p read-only matrix view of the slice.
 *
 * @throws std::invalid_argument if the tensor's dimensions or number of
 * entries are invalid, or if it has more than three dimensions.
 * @throws std::out_of_range if the slice index is outside the third dimension.
 * @throws std::length_error if the storage size exceeds the range of
 * Eigen::Index.
 */
Eigen::Map<const Eigen::MatrixXd> FrontalSlice(const Tensor &tensor,
                                               Eigen::Index slice);

} // namespace tensor_compression
