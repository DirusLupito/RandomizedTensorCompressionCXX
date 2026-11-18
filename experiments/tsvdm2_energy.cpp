#include "tsvdm2_energy.hpp"

#include "mode3_tensor_matrix_product.hpp"
#include "reconstruct_from_tsvdm.hpp"
#include "tsvdm2.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace tensor_compression::experiments {

void Tsvdm2EnergyExperiment::RunExperiment() const {
    // Construct the tensor in the transform domain so that we know its
    // spectrum exactly. The signed permutation swaps the slices and changes
    // one sign without introducing rounding at the exact energy boundaries.
    Tensor transformed{{4, 3, 2}, Eigen::VectorXd::Zero(24)};
    FrontalSlice(transformed, 0).diagonal() << 4.0, 2.0, 1.0;
    FrontalSlice(transformed, 1).diagonal() << 3.0, 1.0, 1.0;
    Eigen::MatrixXd transform(2, 2);
    transform << 0.0, -1.0, 1.0, 0.0;
    const Tensor tensor =
        Mode3TensorMatrixProduct(transformed, transform.transpose());
    const double totalEnergy = tensor.values.squaredNorm();
    AsyncParallelForDriver driver(concurrency);

    /**
     * The expected truncation for one row of the energy sweep.
     *
     * FIELDS:
     * - gamma: The requested fraction of the total energy.
     * - ranks: The number of terms expected in each transformed slice.
     * - energy: The sum of squares of the retained terms.
     */
    struct EnergyCase {
        double gamma;
        std::array<Eigen::Index, 2> ranks;
        double energy;
    };

    // The pooled singular values are 4, 3, 2, 1, 1, 1, with total energy 32.
    // At gamma = 0.5 the first term alone gives exactly 16/32, so MATLAB's
    // strict cutoff also keeps the 3. At 0.95 the cutoff reaches the three
    // tied ones, which must all survive even though they span both slices.
    const EnergyCase cases[]{
        {0.25, {1, 0}, 16.0}, {0.50, {1, 1}, 25.0}, {0.75, {1, 1}, 25.0},
        {0.80, {2, 1}, 29.0}, {0.95, {3, 3}, 32.0}, {1.00, {3, 3}, 32.0},
    };
    std::cout << "gamma\tslice ranks\tkept of 6\tretained energy\n";
    for (const EnergyCase &expected : cases) {
        const Tsvdm2Result result =
            Tsvdm2(tensor, transform, driver, expected.gamma);
        if (!std::ranges::equal(result.ranks, expected.ranks)) {
            throw std::runtime_error(
                "star-M SVD II kept the wrong slice ranks at gamma = " +
                std::to_string(expected.gamma));
        }

        // Check what the returned factors actually reconstruct. With an
        // orthogonal transform the residual contains the discarded energy.
        const Tensor reconstructed =
            ReconstructFromTsvdm(result.factors, transform);
        const double retained = reconstructed.values.squaredNorm();
        const double discarded =
            (tensor.values - reconstructed.values).squaredNorm();
        constexpr double tolerance = 1e-12;
        if (!(std::abs(retained - expected.energy) <= tolerance &&
              std::abs(discarded - (totalEnergy - expected.energy)) <=
                  tolerance)) {
            throw std::runtime_error("star-M SVD II reconstruction has "
                                     "the wrong energy at gamma = " +
                                     std::to_string(expected.gamma));
        }
        std::cout << expected.gamma << "\t(" << result.ranks[0] << ", "
                  << result.ranks[1] << ")\t\t"
                  << result.ranks[0] + result.ranks[1] << "\t\t"
                  << 100.0 * retained / totalEnergy << "%\n";
    }
    std::cout << name << " verification passed.\n\n";
}

} // namespace tensor_compression::experiments
