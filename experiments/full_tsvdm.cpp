#include "full_tsvdm.hpp"

#include "experiment_helpers.hpp"
#include "mode3_tensor_matrix_product.hpp"
#include "tsvdm.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tensor_compression::experiments {

void FullTsvdmExperiment::RunExperiment() const {
    AsyncParallelForDriver &driver = GetExperimentDriver();

    // A tall shape (m > p) and a wide shape (m < p), so the full and economy
    // factors have different sizes, and a mix-up between m and p is caught.
    const std::array<std::array<Eigen::Index, 3>, 2> shapes{
        {{6, 4, 5}, {3, 5, 4}}};
    std::mt19937 generator(548);

    // Every check is a relative error, so rounding should keep each one near
    // machine precision for these small tensors.
    constexpr double tolerance = 1e-12;
    for (const auto &[m, p, n] : shapes) {
        const Tensor tensor = RandomTensor({m, p, n}, generator);
        const Eigen::MatrixXd transform = RandomOrthogonal(n, generator);
        const Tensor transformed = Mode3TensorMatrixProduct(tensor, transform);

        for (const bool economy : {false, true}) {
            const TsvdmFactors inTransform =
                Tsvdm(tensor, transform, driver, {}, true, economy);
            const TsvdmFactors inOriginal =
                Tsvdm(tensor, transform, driver, {}, false, economy);

            // Full factors are m x m, m x p, and p x p per slice. Economy
            // factors replace each m or p beside S with min(m, p).
            const Eigen::Index q = std::min(m, p);
            const Eigen::Index uColumns = economy ? q : m;
            const Eigen::Index vColumns = economy ? q : p;
            for (const TsvdmFactors *factors : {&inTransform, &inOriginal}) {
                if (factors->u.dimensions !=
                        std::vector<Eigen::Index>{m, uColumns, n} ||
                    factors->s.dimensions !=
                        std::vector<Eigen::Index>{uColumns, vColumns, n} ||
                    factors->v.dimensions !=
                        std::vector<Eigen::Index>{p, vColumns, n}) {
                    throw std::runtime_error(
                        "star-M SVD factors have the wrong dimensions.");
                }
            }

            // Each transform domain slice must be an ordinary matrix SVD of
            // the corresponding transformed slice.
            double sliceError = 0.0;
            for (Eigen::Index i = 0; i < n; ++i) {
                const auto slice = FrontalSlice(transformed, i);
                const Eigen::MatrixXd product =
                    FrontalSlice(inTransform.u, i) *
                    FrontalSlice(inTransform.s, i) *
                    FrontalSlice(inTransform.v, i).transpose();
                sliceError = std::max(sliceError,
                                      (product - slice).norm() / slice.norm());

                const Eigen::VectorXd singularValues =
                    FrontalSlice(inTransform.s, i).diagonal();
                for (Eigen::Index k = 0; k < singularValues.size(); ++k) {
                    if (singularValues(k) < 0.0 ||
                        (k > 0 && singularValues(k) > singularValues(k - 1))) {
                        throw std::runtime_error(
                            "star-M SVD singular values are not nonnegative "
                            "and descending.");
                    }
                }
            }

            // Multiplying the original domain factors by M must give back the
            // transform domain factors exactly, up to rounding.
            double domainError = 0.0;
            const std::array<std::pair<const Tensor *, const Tensor *>, 3>
                pairs{{{&inOriginal.u, &inTransform.u},
                       {&inOriginal.s, &inTransform.s},
                       {&inOriginal.v, &inTransform.v}}};
            for (const auto &[original, expected] : pairs) {
                const Tensor returned =
                    Mode3TensorMatrixProduct(*original, transform);
                domainError = std::max(
                    domainError, (returned.values - expected->values).norm() /
                                     expected->values.norm());
            }

            std::cout << "star-M SVD of a random " << m << "x" << p << "x" << n
                      << " tensor (" << (economy ? "economy" : "full")
                      << ")\nLargest relative slice error: " << sliceError
                      << "\nLargest relative domain error: " << domainError
                      << "\n";
            if (sliceError > tolerance || domainError > tolerance) {
                throw std::runtime_error("star-M SVD verification failed.");
            }
        }
    }
    std::cout << "star-M SVD verification passed.\n\n";
}

} // namespace tensor_compression::experiments
