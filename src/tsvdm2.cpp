#include "tsvdm2.hpp"

#include "mode3_tensor_matrix_product.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>

namespace tensor_compression {

/**
 * Finds the last singular value needed to meet the global energy target.
 *
 * @param s The untruncated transform-domain S factor.
 * @param gamma The requested fraction of energy, in (0, 1].
 * @return The smallest singular value to keep, or infinity for a zero tensor.
 * @throws std::runtime_error if a singular value is nonfinite.
 * @throws std::length_error if the spectrum is too large to store.
 * @throws std::bad_alloc if there is not enough memory for the spectrum.
 */
static double FindEnergyCutoff(const Tensor &s, double gamma) {
    const Eigen::Index slices = s.dimensions[2];
    const Eigen::Index terms = std::min(s.dimensions[0], s.dimensions[1]);
    std::vector<double> spectrum;
    spectrum.reserve(static_cast<std::size_t>(terms * slices));
    for (Eigen::Index i = 0; i < slices; ++i) {
        const auto slice = FrontalSlice(s, i);
        for (Eigen::Index j = 0; j < terms; ++j) {
            const double value = slice(j, j);
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Nonfinite singular value in star-M SVD II.");
            }
            if (value > 0.0) {
                spectrum.push_back(value);
            }
        }
    }
    if (spectrum.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    // Sorting the positive singular values also orders their squares. Using
    // the singular value itself for the cutoff keeps ties across slices.
    std::ranges::sort(spectrum, std::greater<>{});
    if (gamma == 1.0) {
        return spectrum.back();
    }

    // A common power of two cancels in the energy fraction. Scale before
    // squaring so even a tensor with very large or very small entries can
    // use the same comparison as MATLAB's cumsum(Delta) / w(end) > gamma.
    const int exponent = std::ilogb(spectrum.front());
    const auto energy = [exponent](double value) {
        const double scaled = std::scalbn(value, -exponent);
        return scaled * scaled;
    };
    double total = 0.0;
    for (const double value : spectrum) {
        total += energy(value);
    }
    double retained = 0.0;
    for (const double value : spectrum) {
        retained += energy(value);
        if (retained / total > gamma) {
            return value;
        }
    }
    return spectrum.back();
}

Tsvdm2Result Tsvdm2(const Tensor &tensor, const Eigen::MatrixXd &transform,
                    double gamma, bool transformDomain, bool economy) {
    AsyncParallelForDriver serial(1);
    return Tsvdm2(tensor, transform, serial, gamma, transformDomain, economy);
}

Tsvdm2Result Tsvdm2(const Tensor &tensor, const Eigen::MatrixXd &transform,
                    AsyncParallelForDriver &driver, double gamma,
                    bool transformDomain, bool economy, std::size_t batchSize) {
    if (!(gamma > 0.0 && gamma <= 1.0)) {
        throw std::invalid_argument("star-M SVD II requires gamma in (0, 1].");
    }

    // MATLAB's TSVDM2 calls TSVDM for the slice SVDs. Tsvdm waits for the
    // driver to finish every slice before we pool their singular values.
    Tsvdm2Result result{
        Tsvdm(tensor, transform, driver, {}, true, economy, batchSize), {}};
    TsvdmFactors &factors = result.factors;
    const double cutoff = FindEnergyCutoff(factors.s, gamma);
    const Eigen::Index slices = factors.s.dimensions[2];
    result.ranks.reserve(static_cast<std::size_t>(slices));

    // TSVDM2 leaves U and V at their original sizes. Each S slice already
    // has its singular values in descending order and zeros off the diagonal.
    for (Eigen::Index i = 0; i < slices; ++i) {
        auto slice = FrontalSlice(factors.s, i);
        auto diagonal = slice.diagonal();
        const Eigen::Index rank = (diagonal.array() >= cutoff).count();
        result.ranks.push_back(rank);
        diagonal.tail(diagonal.size() - rank).setZero();
    }
    if (!transformDomain) {
        factors.u = Mode3TensorMatrixProduct(factors.u, transform.transpose());
        factors.s = Mode3TensorMatrixProduct(factors.s, transform.transpose());
        factors.v = Mode3TensorMatrixProduct(factors.v, transform.transpose());
    }
    return result;
}

} // namespace tensor_compression
