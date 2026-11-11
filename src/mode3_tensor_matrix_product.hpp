#pragma once

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace tensor_compression {

/**
 * Selects how to apply the matrix to each mode-3 fiber.
 *
 * MEMBERS:
 * - AsProvided: Apply the matrix as in tmprod(T, M, 3).
 * - Transpose: Apply its plain transpose, corresponding to MATLAB's 'T'.
 * - ConjugateTranspose: Apply its adjoint, corresponding to MATLAB's 'H'.
 */
enum class Mode3MatrixOperation : std::uint8_t {
    AsProvided,
    Transpose,
    ConjugateTranspose,
};

/**
 * Owns the product in column-major order and records how to interpret its axes.
 *
 * @tparam Scalar Scalar type of the product.
 *
 * FIELDS:
 * - values: Contiguous Eigen matrix holding the product. Each column is a
 *   flattened frontal slice in standard order, or a transformed fiber in
 *   mode-first order.
 * - dimensions: The three tensor extents in the order stored by values.
 * - inversePermutation: One-based axes to pass to MATLAB's permute to recover
 *   standard order. This is {2, 3, 1} for mode-first output and the identity
 *   permutation for standard output.
 */
template <typename Scalar> struct Mode3Product {
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
        values;
    std::array<Eigen::Index, 3> dimensions;
    std::array<std::uint32_t, 3> inversePermutation;
};

namespace detail {

/**
 * Multiplies two nonnegative extents without overflowing Eigen's index type.
 *
 * @param left First extent, already validated to be nonnegative.
 * @param right Second extent, already validated to be nonnegative.
 * @return The number of elements in an array with these extents.
 * @throws std::length_error if the element count exceeds Eigen's index range.
 */
inline Eigen::Index CheckedExtentProduct(Eigen::Index left,
                                         Eigen::Index right) {
    if (right != 0 &&
        left > (std::numeric_limits<Eigen::Index>::max)() / right) {
        throw std::length_error(
            "Mode-3 tensor dimensions exceed Eigen's index range.");
    }
    return left * right;
}

} // namespace detail

/**
 * Applies a dense matrix to each fiber along the third axis, as in Tensorlab's
 * tmprod(T, M, 3). The input is viewed in place through Eigen and
 * multiplication is delegated to Eigen's dense matrix API.
 *
 * @tparam TensorElement Element type of the input span, optionally const.
 * @tparam Extent Static extent of the span or std::dynamic_extent.
 * @tparam Derived Dense Eigen matrix expression type. Its scalar must support
 * multiplication with the tensor scalar under Eigen's scalar rules.
 * @param tensorValues Tensor elements in MATLAB column-major order, with the
 * first index varying fastest. The function borrows this span for the call.
 * @param dimensions Tensor shape {m, p, n}. Use n = 1 for a matrix input.
 * @param matrix Matrix to apply. Its effective shape after operation must be
 * r by n. It can be rectangular.
 * @param operation Optional transpose or conjugate transpose of matrix.
 * @param savePermutation Return the mode-first layout used by tmprod's
 * saveperm option and its two-output form.
 * @return Owning product with shape {m, p, r}, or {r, m, p} when
 * savePermutation is true. The result includes the inverse permutation.
 * @throws std::invalid_argument for negative extents, a buffer or matrix shape
 * mismatch, or an invalid operation.
 * @throws std::length_error if an element count exceeds Eigen's index range.
 * @throws std::bad_alloc if Eigen cannot allocate the result or workspace.
 */
template <typename TensorElement, std::size_t Extent, typename Derived>
auto Mode3TensorMatrixProduct(
    std::span<TensorElement, Extent> tensorValues,
    const std::array<Eigen::Index, 3> &dimensions,
    const Eigen::MatrixBase<Derived> &matrix,
    Mode3MatrixOperation operation = Mode3MatrixOperation::AsProvided,
    bool savePermutation = false) {
    using TensorScalar = std::remove_const_t<TensorElement>;
    using Scalar = typename Eigen::ScalarBinaryOpTraits<
        TensorScalar, typename Derived::Scalar>::ReturnType;
    using InputMatrix = Eigen::Matrix<TensorScalar, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::ColMajor>;

    const auto [rows, columns, inputDepth] = dimensions;
    if (rows < 0 || columns < 0 || inputDepth < 0) {
        throw std::invalid_argument(
            "Mode-3 tensor dimensions must be nonnegative.");
    }
    const Eigen::Index fiberCount = detail::CheckedExtentProduct(rows, columns);
    const Eigen::Index inputSize =
        detail::CheckedExtentProduct(fiberCount, inputDepth);
    if (tensorValues.size() != static_cast<std::size_t>(inputSize)) {
        throw std::invalid_argument(
            "Mode-3 tensor buffer size does not match its dimensions.");
    }

    // Each column of this view is one frontal slice. Each row is one fiber, so
    // a right multiply by the effective matrix's transpose transforms all
    // fibers.
    const Eigen::Map<const InputMatrix> slices(tensorValues.data(), fiberCount,
                                               inputDepth);

    // Keep shape validation and output layout the same for all matrix
    // operations.
    const auto applyMatrix = [&](const auto &effectiveMatrix) {
        if (effectiveMatrix.cols() != inputDepth) {
            throw std::invalid_argument("Mode-3 matrix input width must match "
                                        "the tensor's third dimension.");
        }
        const Eigen::Index outputDepth = effectiveMatrix.rows();
        const Eigen::Index outputSize =
            detail::CheckedExtentProduct(fiberCount, outputDepth);
        Mode3Product<Scalar> result;
        if (savePermutation) {
            result.values.resize(outputDepth, fiberCount);
            result.dimensions = {outputDepth, rows, columns};
            result.inversePermutation = {2, 3, 1};
        } else {
            result.values.resize(fiberCount, outputDepth);
            result.dimensions = {rows, columns, outputDepth};
            result.inversePermutation = {1, 2, 3};
        }

        // Preserve the shape of empty outputs and treat an empty fiber's dot
        // product as zero without accessing the input buffer.
        if (outputSize == 0) {
            return result;
        }
        if (inputDepth == 0) {
            result.values.setZero();
        } else if (savePermutation) {
            result.values.noalias() = effectiveMatrix * slices.transpose();
        } else {
            result.values.noalias() = slices * effectiveMatrix.transpose();
        }
        return result;
    };

    switch (operation) {
    case Mode3MatrixOperation::AsProvided:
        return applyMatrix(matrix.derived());
    case Mode3MatrixOperation::Transpose:
        return applyMatrix(matrix.transpose());
    case Mode3MatrixOperation::ConjugateTranspose:
        return applyMatrix(matrix.adjoint());
    }
    throw std::invalid_argument("Invalid mode-3 matrix operation.");
}

} // namespace tensor_compression
