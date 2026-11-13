#include "permute.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace tensor_compression {

Tensor Permute(const Tensor &tensor, const std::vector<Eigen::Index> &axes) {
    ValidateTensor(tensor);
    if (axes.size() < tensor.dimensions.size()) {
        throw std::invalid_argument(
            "A permutation must include every tensor axis.");
    }

    // The public axis numbers match MATLAB. Convert them to vector indices
    // after checking the range, and reject repeated axes before using them
    // to rearrange any entries.
    std::vector<std::size_t> sourceAxes;
    for (const Eigen::Index mode : axes) {
        if (mode < 1 || static_cast<std::size_t>(mode) > axes.size()) {
            throw std::invalid_argument(
                "Permutation axes must be between 1 and the number of axes.");
        }
        const std::size_t axis = static_cast<std::size_t>(mode - 1);
        if (std::find(sourceAxes.begin(), sourceAxes.end(), axis) !=
            sourceAxes.end()) {
            throw std::invalid_argument("Permutation axes must be distinct.");
        }
        sourceAxes.push_back(axis);
    }

    Tensor result;
    for (const std::size_t axis : sourceAxes) {
        result.dimensions.push_back(
            axis < tensor.dimensions.size() ? tensor.dimensions[axis] : 1);
    }

    // Each axis appears exactly once, so sorted indices give the original
    // axis order. The entries are already in the right order and a copy is
    // enough.
    if (std::is_sorted(sourceAxes.begin(), sourceAxes.end())) {
        result.values = tensor.values;
        return result;
    }
    result.values.resize(tensor.values.size());
    if (tensor.values.size() == 0) {
        return result;
    }

    // For an m x p x n tensor, moving one position along the first, second,
    // or third axis advances 1, m, or m*p entries in values, respectively.
    // These offsets are the strides used to find an entry from its coordinates.
    std::vector<Eigen::Index> strides(sourceAxes.size(), 1);
    for (std::size_t axis = 1; axis < sourceAxes.size(); ++axis) {
        const Eigen::Index previous =
            axis <= tensor.dimensions.size() ? tensor.dimensions[axis - 1] : 1;
        strides[axis] = strides[axis - 1] * previous;
    }

    // We fill the result in storage order. Taking the remainder gives the
    // coordinate along the current axis, and dividing removes that coordinate
    // so we can recover the next one. sourceAxes tells us which original stride
    // to use for each coordinate, giving the position of the entry to copy.
    for (Eigen::Index output = 0; output < result.values.size(); ++output) {
        Eigen::Index remaining = output;
        Eigen::Index input = 0;
        for (std::size_t axis = 0; axis < sourceAxes.size(); ++axis) {
            input += (remaining % result.dimensions[axis]) *
                     strides[sourceAxes[axis]];
            remaining /= result.dimensions[axis];
        }
        result.values[output] = tensor.values[input];
    }
    return result;
}

} // namespace tensor_compression
