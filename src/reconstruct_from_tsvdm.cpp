#include "reconstruct_from_tsvdm.hpp"

#include "mode3_tensor_matrix_product.hpp"

#include <stdexcept>
#include <vector>

namespace tensor_compression {

Tensor ReconstructFromTsvdm(const TsvdmFactors &factors,
                            const Eigen::MatrixXd &transform) {
    ValidateTsvdmFactors(factors);
    const Eigen::Index m = factors.u.dimensions[0];
    const Eigen::Index p = factors.v.dimensions[0];
    const Eigen::Index n = factors.s.dimensions[2];
    if (transform.rows() != n || transform.cols() != n) {
        throw std::invalid_argument(
            "star-M SVD transform must be n x n for factors with n slices.");
    }

    // Move every factor to the transform domain, where the star-M product
    // becomes an ordinary matrix product on each frontal slice.
    const Tensor uHat = Mode3TensorMatrixProduct(factors.u, transform);
    const Tensor sHat = Mode3TensorMatrixProduct(factors.s, transform);
    const Tensor vHat = Mode3TensorMatrixProduct(factors.v, transform);

    // Multiply the factors slice by slice. Each slice is independent, as in
    // Tsvdm's decomposition loop.
    const std::vector<Eigen::Index> dimensions{m, p, n};
    Tensor product{dimensions, Eigen::VectorXd(ElementCount(dimensions))};
    for (Eigen::Index i = 0; i < n; ++i) {
        FrontalSlice(product, i) = FrontalSlice(uHat, i) *
                                   FrontalSlice(sHat, i) *
                                   FrontalSlice(vHat, i).transpose();
    }

    // Return to the original domain. Since M is orthogonal, its transpose is
    // its inverse.
    return Mode3TensorMatrixProduct(product, transform.transpose());
}

} // namespace tensor_compression
