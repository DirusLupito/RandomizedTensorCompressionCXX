#include "identity_tsvdm.hpp"

#include "experiment_helpers.hpp"
#include "tsvdm.hpp"
#include <Eigen/SVD>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace tensor_compression::experiments {

void IdentityTsvdmExperiment::RunExperiment() const {
    AsyncParallelForDriver &driver = GetExperimentDriver();

    const std::array<std::array<Eigen::Index, 3>, 2> shapes{
        {{6, 4, 5}, {3, 5, 4}}};
    std::mt19937 generator(548);

    // Multiplying by the identity is exact in floating point, so any
    // difference beyond rounding means the slices were mixed up.
    constexpr double tolerance = 1e-12;
    for (const auto &[m, p, n] : shapes) {
        const Tensor tensor = RandomTensor({m, p, n}, generator);
        const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n, n);
        for (const bool economy : {false, true}) {
            const TsvdmFactors factors =
                Tsvdm(tensor, identity, driver, {}, false, economy);
            const std::uint32_t options =
                economy ? (Eigen::ComputeThinU | Eigen::ComputeThinV)
                        : (Eigen::ComputeFullU | Eigen::ComputeFullV);
            double error = 0.0;
            for (Eigen::Index i = 0; i < n; ++i) {
                const Eigen::BDCSVD<Eigen::MatrixXd> svd(
                    FrontalSlice(tensor, i), options);
                const Eigen::MatrixXd s = FrontalSlice(factors.s, i);
                Eigen::MatrixXd expectedS =
                    Eigen::MatrixXd::Zero(s.rows(), s.cols());
                expectedS.diagonal() = svd.singularValues();
                error = std::max(
                    {error, (FrontalSlice(factors.u, i) - svd.matrixU()).norm(),
                     (s - expectedS).norm(),
                     (FrontalSlice(factors.v, i) - svd.matrixV()).norm()});
            }
            std::cout << "Identity-transform star-M SVD of a random " << m
                      << "x" << p << "x" << n << " tensor ("
                      << (economy ? "economy" : "full")
                      << ")\nLargest difference from the slice SVDs: " << error
                      << "\n";
            if (error > tolerance) {
                throw std::runtime_error(
                    "Identity-transform star-M SVD verification failed.");
            }
        }
    }
    std::cout << "Identity-transform star-M SVD verification passed.\n\n";
}

} // namespace tensor_compression::experiments
