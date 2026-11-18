#include "reconstructed_tsvdm.hpp"

#include "experiment_helpers.hpp"
#include "reconstruct_from_tsvdm.hpp"
#include <array>
#include <iostream>
#include <stdexcept>

namespace tensor_compression::experiments {

void ReconstructedTsvdmExperiment::RunExperiment() const {
    AsyncParallelForDriver &driver = GetExperimentDriver();

    // The same tall and wide shapes as the full SVD experiment, so a mix-up
    // between m and p in the reconstruction would be caught.
    const std::array<std::array<Eigen::Index, 3>, 2> shapes{
        {{6, 4, 5}, {3, 5, 4}}};
    std::mt19937 generator(548);

    // The relative Frobenius error. The teammate set this tolerance. It is
    // far looser than the rounding error, so it only fails on a real bug.
    constexpr double tolerance = 1e-6;
    for (const auto &[m, p, n] : shapes) {
        const Tensor tensor = RandomTensor({m, p, n}, generator);
        const Eigen::MatrixXd transform = RandomOrthogonal(n, generator);
        for (const bool economy : {false, true}) {
            const TsvdmFactors factors =
                Tsvdm(tensor, transform, driver, {}, false, economy);
            const Tensor reconstructed =
                ReconstructFromTsvdm(factors, transform);
            if (reconstructed.dimensions != tensor.dimensions) {
                throw std::runtime_error(
                    "Reconstructed tensor has the wrong dimensions.");
            }
            const double error = (reconstructed.values - tensor.values).norm() /
                                 tensor.values.norm();
            std::cout << "Reconstruction of a random " << m << "x" << p << "x"
                      << n << " tensor (" << (economy ? "economy" : "full")
                      << ")\nRelative error: " << error << "\n";
            if (error > tolerance) {
                throw std::runtime_error(
                    "star-M SVD reconstruction verification failed.");
            }
        }
    }
    std::cout << "star-M SVD reconstruction verification passed.\n\n";
}

} // namespace tensor_compression::experiments
