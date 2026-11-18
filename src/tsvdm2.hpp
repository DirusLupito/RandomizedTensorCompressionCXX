#pragma once

#include "tsvdm.hpp"

#include <cstddef>
#include <vector>

namespace tensor_compression {

/**
 * Holds the result of choosing the star-M SVD ranks by retained energy.
 *
 * FIELDS:
 * - factors: The full or economy factors U, S, and V. Discarded singular
 * values are zero in the transform domain, as in MATLAB's TSVDM2.
 * - ranks: The number of singular values kept in each transformed frontal
 * slice, corresponding to MATLAB's rho.
 */
struct Tsvdm2Result {
    TsvdmFactors factors;
    std::vector<Eigen::Index> ranks;
};

/**
 * Computes the serial star-M SVD II. All slices contribute to one energy
 * budget, so a slice with large singular values can keep more terms than a
 * slice with small ones. Energy is the sum of squared singular values.
 *
 * @param tensor The real m x p x n tensor to decompose. As in Tsvdm, a
 * two-dimensional tensor has one frontal slice. All extents must be positive.
 * @param transform The orthogonal n x n matrix M defining the star-M product.
 * @param gamma The fraction of total energy to retain, in (0, 1]. For a
 * nonzero tensor and gamma below 1, keep terms until their combined energy
 * fraction first exceeds gamma, including all ties at the cutoff. At 1,
 * keep every positive singular value. A zero tensor has zero ranks throughout.
 * @param transformDomain Whether to return the factors in the transform
 * domain. Defaults to false, which returns them in the original domain.
 * @param economy Whether to compute economy-sized factors. Defaults to false.
 *
 * @return The factors and their per-slice ranks. Truncation preserves the
 * factor dimensions for the selected full or economy form.
 *
 * @throws std::invalid_argument if gamma is outside (0, 1] or is NaN, or
 * if the tensor or transform dimensions are invalid.
 * @throws std::runtime_error if a slice SVD fails or produces nonfinite
 * singular values.
 * @throws std::length_error if the required storage size cannot be represented.
 * @throws std::bad_alloc if there is not enough memory for the decomposition.
 */
Tsvdm2Result Tsvdm2(const Tensor &tensor, const Eigen::MatrixXd &transform,
                    double gamma = 1.0, bool transformDomain = false,
                    bool economy = false);

/**
 * Computes star-M SVD II with the slice SVDs scheduled on the given driver.
 * This corresponds to TSVDM2 calling TSVDM in MATLAB, where parfor runs the
 * independent slice SVDs. Once all slices finish, the calling thread selects
 * a global energy cutoff and applies it to each slice.
 *
 * @param tensor The input tensor, with the same dimensions as the serial
 * overload.
 * @param transform The orthogonal n x n matrix M defining the star-M product.
 * @param driver The caller's reusable slice SVD driver. A concurrency of one
 * runs on the calling thread. See Tsvdm for the driver's requirements.
 * @param gamma The fraction of total energy to retain, in (0, 1]. Defaults
 * to 1.
 * @param transformDomain Whether to keep the factors in the transform domain.
 * Defaults to false, which returns them in the original domain.
 * @param economy Whether to compute economy-sized factors. Defaults to false.
 * @param batchSize The number of consecutive slices claimed at a time.
 * Defaults to 1.
 *
 * @return The factors and per-slice ranks, with the same cutoff and treatment
 * of ties as the serial overload.
 * @throws std::invalid_argument if the input dimensions or gamma are invalid,
 * or if batchSize is zero.
 * @throws std::runtime_error if a slice SVD fails or produces nonfinite values.
 * @throws std::length_error if the required storage size cannot be represented.
 * @throws std::bad_alloc if there is not enough memory for the decomposition.
 */
Tsvdm2Result Tsvdm2(const Tensor &tensor, const Eigen::MatrixXd &transform,
                    AsyncParallelForDriver &driver, double gamma = 1.0,
                    bool transformDomain = false, bool economy = false,
                    std::size_t batchSize = 1);

} // namespace tensor_compression
