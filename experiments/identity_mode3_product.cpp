#include "identity_mode3_product.hpp"

#include "mode3_tensor_matrix_product.hpp"
#include <iostream>
#include <random>
#include <stdexcept>

namespace tensor_compression::experiments {

void IdentityMode3ProductExperiment::RunExperiment() const {
    // Use different sizes for each dimension so an incorrect permutation is
    // easier to spot. The fixed seed lets us repeat the same calculation.
    constexpr Eigen::Index m = 2;
    constexpr Eigen::Index p = 3;
    constexpr Eigen::Index n = 4;
    Tensor tensor{{m, p, n}, Eigen::VectorXd(m * p * n)};
    std::mt19937 generator(581);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (Eigen::Index i = 0; i < tensor.values.size(); ++i) {
        tensor.values(i) = distribution(generator);
    }

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n, n);
    const Tensor result = Mode3TensorMatrixProduct(tensor, identity);
    if (result.dimensions != tensor.dimensions ||
        result.values.size() != tensor.values.size()) {
        throw std::runtime_error(
            "Identity mode-3 product changed the tensor's dimensions.");
    }

    // The vector contains every tensor entry, so its Euclidean norm is the
    // tensor's Frobenius norm. Equal norms alone do not establish equality:
    // reordering entries preserves the norm, so also measure the difference.
    const double inputNorm = tensor.values.norm();
    const double outputNorm = result.values.norm();
    const double differenceNorm = (result.values - tensor.values).norm();
    std::cout << "Mode-3 product of a random " << m << "x" << p << "x" << n
              << " tensor with the " << n << "x" << n << " identity matrix\n"
              << "Input Frobenius norm: " << inputNorm
              << "\nOutput Frobenius norm: " << outputNorm
              << "\nFrobenius norm of the difference: " << differenceNorm
              << "\n\nElementwise comparison (one-based indices):\n";

    // Inspect every corresponding pair using MATLAB's column-major ordering.
    // Print the indices starting at one to make comparison with MATLAB easier.
    Eigen::Index mismatches = 0;
    for (Eigen::Index k = 0; k < n; ++k) {
        for (Eigen::Index j = 0; j < p; ++j) {
            for (Eigen::Index i = 0; i < m; ++i) {
                const Eigen::Index index = i + m * (j + p * k);
                const bool equal = tensor.values(index) == result.values(index);
                if (!equal) {
                    ++mismatches;
                }
                std::cout << "(" << i + 1 << ", " << j + 1 << ", " << k + 1
                          << "): " << tensor.values(index) << " -> "
                          << result.values(index)
                          << (equal ? " (equal)\n" : " (mismatch)\n");
            }
        }
    }
    std::cout << "Elementwise mismatches: " << mismatches << "\n";

    // The identity only multiplies these finite entries by zero and one, so
    // the result should preserve them exactly, without a rounding tolerance.
    if (inputNorm != outputNorm || differenceNorm != 0.0 || mismatches != 0) {
        throw std::runtime_error(
            "Identity mode-3 product verification failed.");
    }
    std::cout << "Identity mode-3 product verification passed.\n\n";
}

} // namespace tensor_compression::experiments
