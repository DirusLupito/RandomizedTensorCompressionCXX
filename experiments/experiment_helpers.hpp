#pragma once

#include "async_parallel_for_driver.hpp"
#include "tensor.hpp"

#include <random>
#include <vector>

namespace tensor_compression::experiments {

/**
 * Creates a tensor with entries drawn uniformly from [-1, 1], so the checks
 * run on generic data with no special structure.
 *
 * @param dimensions The size of the tensor along each axis. Each size must be
 * nonnegative.
 * @param generator The random number generator to draw from. A seeded
 * generator makes the checks repeatable. Drawing advances its state.
 *
 * @return The random tensor.
 *
 * @throws std::invalid_argument if a dimension is negative.
 * @throws std::length_error if the storage size cannot be represented.
 * @throws std::bad_alloc if there is not enough memory for the tensor.
 */
Tensor RandomTensor(const std::vector<Eigen::Index> &dimensions,
                    std::mt19937 &generator);

/**
 * Creates a random orthogonal matrix to use as a star-M transform. The Q factor
 * of a random matrix's QR decomposition is orthogonal, which makes a valid
 * transform with no special structure, unlike the identity or a DCT.
 *
 * @param size The number of rows and columns, n. Must be nonnegative.
 * @param generator The random number generator to draw from. Drawing advances
 * its state.
 *
 * @return An n x n orthogonal matrix.
 *
 * @throws std::bad_alloc if there is not enough memory for the matrix or
 * temporary data.
 */
Eigen::MatrixXd RandomOrthogonal(Eigen::Index size, std::mt19937 &generator);

/**
 * Checks whether two tensors have exactly the same dimensions and entries,
 * with no rounding tolerance.
 *
 * @param left The first tensor to compare.
 * @param right The second tensor to compare.
 *
 * @return true if the tensors are identical, false otherwise.
 */
bool SameTensor(const Tensor &left, const Tensor &right);

/**
 * @return The worker pool shared by the SVD experiments. The first call
 * starts the workers, and subsequent experiments reuse them.
 * @throws std::system_error if a worker cannot be started.
 * @throws std::bad_alloc if there is not enough memory for the workers.
 */
AsyncParallelForDriver &GetExperimentDriver();

} // namespace tensor_compression::experiments
