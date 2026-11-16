#include "tensor.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tensor_compression {

Eigen::Index ElementCount(std::span<const Eigen::Index> dimensions) {
    for (const Eigen::Index dimension : dimensions) {
        if (dimension < 0) {
            throw std::invalid_argument(
                "Tensor dimensions must be nonnegative.");
        }
    }

    // If any dimension is zero, there are no entries, regardless of how large
    // the other dimensions are. Check this before multiplying their sizes.
    if (std::find(dimensions.begin(), dimensions.end(), 0) !=
        dimensions.end()) {
        return 0;
    }

    // Each entry occupies sizeof(double) bytes. We check against the limit
    // before multiplying so the multiplication itself cannot overflow.
    const Eigen::Index limit =
        (std::numeric_limits<Eigen::Index>::max)() / sizeof(double);
    Eigen::Index count = 1;
    for (const Eigen::Index dimension : dimensions) {
        if (count > limit / dimension) {
            throw std::length_error(
                "Tensor storage exceeds Eigen's index range.");
        }
        count *= dimension;
    }
    return count;
}

void ValidateTensor(const Tensor &tensor) {
    if (tensor.dimensions.size() < 2 ||
        ElementCount(tensor.dimensions) != tensor.values.size()) {
        throw std::invalid_argument(
            "Tensor data must match a shape with at least two axes.");
    }
}

/**
 * Finds where a frontal slice begins in a tensor's data vector, checking that
 * the tensor can be viewed slice by slice and that the slice exists. Both
 * FrontalSlice overloads share this so their checks cannot drift apart.
 *
 * @param tensor The tensor containing the slice. It must have two or three
 * dimensions, and its entries must agree with them.
 * @param slice The zero-based index of the frontal slice.
 *
 * @return The index of the slice's first entry in the data vector.
 *
 * @throws std::invalid_argument if the tensor's dimensions or number of
 * entries are invalid, or if it has more than three dimensions.
 * @throws std::out_of_range if the slice index is outside the third dimension.
 * @throws std::length_error if the storage size exceeds the range of
 * Eigen::Index.
 */
static Eigen::Index FrontalSliceOffset(const Tensor &tensor,
                                       Eigen::Index slice) {
    ValidateTensor(tensor);
    if (tensor.dimensions.size() > 3) {
        throw std::invalid_argument(
            "Frontal slices require a tensor with at most three axes.");
    }

    // A two-dimensional tensor is a single frontal slice, as in MATLAB.
    const Eigen::Index slices =
        tensor.dimensions.size() == 3 ? tensor.dimensions[2] : 1;
    if (slice < 0 || slice >= slices) {
        throw std::out_of_range("Frontal slice index is out of range.");
    }
    return slice * tensor.dimensions[0] * tensor.dimensions[1];
}

Eigen::Map<Eigen::MatrixXd> FrontalSlice(Tensor &tensor, Eigen::Index slice) {
    const Eigen::Index offset = FrontalSliceOffset(tensor, slice);
    return Eigen::Map<Eigen::MatrixXd>(tensor.values.data() + offset,
                                       tensor.dimensions[0],
                                       tensor.dimensions[1]);
}

Eigen::Map<const Eigen::MatrixXd> FrontalSlice(const Tensor &tensor,
                                               Eigen::Index slice) {
    const Eigen::Index offset = FrontalSliceOffset(tensor, slice);
    return Eigen::Map<const Eigen::MatrixXd>(tensor.values.data() + offset,
                                             tensor.dimensions[0],
                                             tensor.dimensions[1]);
}

} // namespace tensor_compression
