#include "tmprod.hpp"

#include "permute.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace tensor_compression {

Tensor Tmprod(const Tensor &tensor, const Eigen::MatrixXd &matrix,
              Eigen::Index mode, char transpose, bool savePermutation,
              std::vector<Eigen::Index> *inversePermutation) {
    return Tmprod(tensor, std::span(&matrix, 1), std::span(&mode, 1), transpose,
                  savePermutation, inversePermutation);
}

Tensor Tmprod(const Tensor &tensor, std::span<const Eigen::MatrixXd> matrices,
              std::span<const Eigen::Index> modes, char transpose,
              bool savePermutation,
              std::vector<Eigen::Index> *inversePermutation) {
    ValidateTensor(tensor);
    if (transpose != 'N' && transpose != 'T' && transpose != 'H') {
        throw std::invalid_argument(
            "tmprod transpose must be 'N', 'T', or 'H'.");
    }
    if (matrices.size() != modes.size()) {
        throw std::invalid_argument("tmprod needs one matrix per mode.");
    }

    // Check every matrix against the original tensor before multiplying any
    // entries. Since the modes are distinct, multiplying along one mode does
    // not change the size that another matrix needs to match. Missing trailing
    // axes have size one, as in MATLAB.
    std::vector<Eigen::Index> dimensions = tensor.dimensions;
    std::vector<double> ratios;
    for (std::size_t i = 0; i < modes.size(); ++i) {
        if (modes[i] < 1 || std::find(modes.begin(), modes.begin() + i,
                                      modes[i]) != modes.begin() + i) {
            throw std::invalid_argument(
                "tmprod modes must be positive and distinct.");
        }
        const std::size_t axis = static_cast<std::size_t>(modes[i] - 1);
        if (axis >= dimensions.size()) {
            dimensions.resize(axis + 1, 1);
        }
        const Eigen::Index columns =
            transpose == 'N' ? matrices[i].cols() : matrices[i].rows();
        const Eigen::Index rows =
            transpose == 'N' ? matrices[i].rows() : matrices[i].cols();
        if (columns != dimensions[axis]) {
            throw std::invalid_argument(
                "tmprod matrix width must match its tensor mode.");
        }

        // Tensorlab sorts the ratios of input size to output size in ascending
        // order. MATLAB represents a positive size divided by zero as infinity,
        // and 0/0 as NaN. Keep those cases so the ordering agrees with MATLAB.
        if (rows != 0) {
            ratios.push_back(static_cast<double>(columns) /
                             static_cast<double>(rows));
        } else if (columns != 0) {
            ratios.push_back(std::numeric_limits<double>::infinity());
        } else {
            ratios.push_back(std::numeric_limits<double>::quiet_NaN());
        }
    }

    // Sort the indices so each matrix stays paired with its mode. Equal ratios
    // keep their original order, and NaN ratios go after all other ratios.
    std::vector<std::size_t> order(modes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t left, std::size_t right) {
                         if (std::isnan(ratios[left])) {
                             return false;
                         }
                         if (std::isnan(ratios[right])) {
                             return true;
                         }
                         return ratios[left] < ratios[right];
                     });

    // Put the selected modes first, in the order we will multiply along them,
    // followed by the remaining axes. For example, if we will multiply along
    // modes 3 and then 1 of a three-dimensional tensor, axes is {3, 1, 2}.
    // We keep track of these original axis numbers so we can restore their
    // order once all the multiplications are complete.
    std::vector<Eigen::Index> axes;
    for (const std::size_t i : order) {
        axes.push_back(modes[i]);
    }
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
        const Eigen::Index mode = static_cast<Eigen::Index>(axis) + 1;
        if (std::find(axes.begin(), axes.end(), mode) == axes.end()) {
            axes.push_back(mode);
        }
    }
    Tensor working = Permute(tensor, axes);

    // After each product except the last, we move the first axis to the end.
    // This brings the next selected mode to the front, where we can multiply
    // it by its matrix in exactly the same way.
    std::vector<Eigen::Index> rotation(axes.size());
    std::iota(rotation.begin(), rotation.end(), Eigen::Index{1});
    std::rotate(rotation.begin(), rotation.begin() + 1, rotation.end());
    for (std::size_t step = 0; step < order.size(); ++step) {
        const Eigen::MatrixXd &matrix = matrices[order[step]];
        const Eigen::Index inputRows = working.dimensions.front();
        Tensor product;
        product.dimensions = working.dimensions;
        product.dimensions.front() =
            transpose == 'N' ? matrix.rows() : matrix.cols();
        product.values.resize(ElementCount(product.dimensions));

        // An empty result has no entries to compute. If the dimension we are
        // summing over is empty but the result is not, each dot product has no
        // terms, so every resulting entry is zero.
        if (product.values.size() != 0) {
            if (inputRows == 0) {
                product.values.setZero();
            } else {
                // With the selected mode first, each column contains one
                // vector to multiply by the matrix. Eigen::Map lets us view
                // the existing entries as a matrix, corresponding to MATLAB's
                // reshape. The input and output use separate memory, so
                // noalias tells Eigen that writing the result cannot overwrite
                // any entries it still needs to read.
                const Eigen::Index columns =
                    product.values.size() / product.dimensions.front();
                const Eigen::Map<const Eigen::MatrixXd> input(
                    working.values.data(), inputRows, columns);
                Eigen::Map<Eigen::MatrixXd> output(
                    product.values.data(), product.dimensions.front(), columns);
                if (transpose == 'N') {
                    output.noalias() = matrix * input;
                } else {
                    output.noalias() = matrix.transpose() * input;
                }
            }
        }
        working = std::move(product);
        if (step + 1 < order.size()) {
            working = Permute(working, rotation);
            std::rotate(axes.begin(), axes.begin() + 1, axes.end());
        }
    }

    // axes tells us which original axis is currently at each position.
    // Reversing that mapping tells us where to find each axis when restoring
    // the original order. For a mode-3 product of a three-dimensional tensor,
    // axes is {3, 1, 2}, so restore is {2, 3, 1}. Both use MATLAB's one-based
    // axis numbers, which can be passed directly to Permute.
    std::vector<Eigen::Index> restore(axes.size());
    for (std::size_t axis = 0; axis < axes.size(); ++axis) {
        restore[static_cast<std::size_t>(axes[axis] - 1)] =
            static_cast<Eigen::Index>(axis) + 1;
    }

    // Requesting MATLAB's second output also skips the final permutation,
    // even if savePermutation is false. The caller can apply it later.
    if (inversePermutation != nullptr) {
        *inversePermutation = std::move(restore);
    } else if (!savePermutation &&
               !std::is_sorted(restore.begin(), restore.end())) {
        working = Permute(working, restore);
    }
    return working;
}

} // namespace tensor_compression
