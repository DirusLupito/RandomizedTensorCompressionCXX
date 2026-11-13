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

} // namespace tensor_compression
