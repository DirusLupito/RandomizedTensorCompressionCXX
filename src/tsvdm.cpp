#include "tsvdm.hpp"

#include "mode3_tensor_matrix_product.hpp"
#include "truncate_tsvdm.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tensor_compression {

void ValidateTsvdmFactors(const TsvdmFactors &factors) {
    for (const Tensor *factor : {&factors.u, &factors.s, &factors.v}) {
        ValidateTensor(*factor);
        if (factor->dimensions.size() != 3) {
            throw std::invalid_argument(
                "star-M SVD factors must each have exactly three axes.");
        }
    }
    const std::vector<Eigen::Index> &u = factors.u.dimensions;
    const std::vector<Eigen::Index> &s = factors.s.dimensions;
    const std::vector<Eigen::Index> &v = factors.v.dimensions;
    if (u[1] != s[0] || v[1] != s[1]) {
        throw std::invalid_argument(
            "star-M SVD factor shapes do not fit together: U must have as many "
            "columns as S has rows, and V as many columns as S has columns.");
    }
    if (u[2] != s[2] || v[2] != s[2]) {
        throw std::invalid_argument(
            "star-M SVD factors must have the same number of frontal slices.");
    }
}

TsvdmFactors Tsvdm(const Tensor &tensor, const Eigen::MatrixXd &transform,
                   AsyncParallelForDriver &driver,
                   std::span<const Eigen::Index> ranks, bool transformDomain,
                   bool economy, std::size_t batchSize) {
    ValidateTensor(tensor);
    if (tensor.dimensions.size() > 3) {
        throw std::invalid_argument(
            "star-M SVD requires a tensor with at most three axes.");
    }

    // A two-dimensional tensor has a single frontal slice, as in MATLAB.
    const Eigen::Index m = tensor.dimensions[0];
    const Eigen::Index p = tensor.dimensions[1];
    const Eigen::Index n =
        tensor.dimensions.size() == 3 ? tensor.dimensions[2] : 1;
    if (m == 0 || p == 0 || n == 0) {
        throw std::invalid_argument(
            "star-M SVD requires a tensor with no empty dimensions.");
    }
    if (transform.rows() != n || transform.cols() != n) {
        throw std::invalid_argument(
            "star-M SVD transform must be n x n for an m x p x n tensor.");
    }

    // Move to the transform domain, where the frontal slices are independent.
    const Tensor transformed = Mode3TensorMatrixProduct(tensor, transform);

    // Allocate the factors with MATLAB's shapes. Full factors have square U
    // and V slices and rectangular S slices. Economy factors drop the columns
    // of U or V that only multiply the zero rows or columns of S. S starts at
    // zero because only its diagonal is filled in below.
    const Eigen::Index singularValueCount = std::min(m, p);
    const Eigen::Index uColumns = economy ? singularValueCount : m;
    const Eigen::Index vColumns = economy ? singularValueCount : p;
    TsvdmFactors factors;
    factors.u.dimensions = {m, uColumns, n};
    factors.u.values.resize(ElementCount(factors.u.dimensions));
    factors.s.dimensions = {uColumns, vColumns, n};
    factors.s.values.setZero(ElementCount(factors.s.dimensions));
    factors.v.dimensions = {p, vColumns, n};
    factors.v.values.resize(ElementCount(factors.v.dimensions));

    // Decompose each transformed slice on its own, in parallel. This is the
    // loop MATLAB's TSVDM runs with parfor. No locking is needed, because
    // slice i only reads its own slice of the transformed tensor and only
    // writes its own slices of U, S, and V, which are separate contiguous
    // blocks of memory. The factors were allocated above, so no thread
    // resizes them. BDCSVD uses the divide-and-conquer approach, like the
    // LAPACK routine behind MATLAB's svd, and returns the singular values in
    // descending order.
    const std::uint32_t options =
        economy ? (Eigen::ComputeThinU | Eigen::ComputeThinV)
                : (Eigen::ComputeFullU | Eigen::ComputeFullV);
    driver.ParallelFor(
        static_cast<std::size_t>(n), batchSize,
        [&transformed, &factors, options](std::size_t begin, std::size_t end,
                                          std::uint32_t) {
            for (std::size_t slice = begin; slice < end; ++slice) {
                const Eigen::Index i = static_cast<Eigen::Index>(slice);
                const Eigen::BDCSVD<Eigen::MatrixXd> svd(
                    FrontalSlice(transformed, i), options);
                if (svd.info() != Eigen::Success) {
                    throw std::runtime_error(
                        "Eigen failed to compute a frontal slice SVD.");
                }
                FrontalSlice(factors.u, i) = svd.matrixU();
                FrontalSlice(factors.s, i).diagonal() = svd.singularValues();
                FrontalSlice(factors.v, i) = svd.matrixV();
            }
        });

    // Only truncate if ranks are given. Truncation happens in the transform
    // domain, where each slice's leading triples give its best approximation.
    if (!ranks.empty()) {
        factors = TruncateTsvdm(factors, ranks);
    }

    // If we are told to return in the transform domain, do so here.
    if (transformDomain) {
        return factors;
    }

    // Return to the original domain. Since M is orthogonal, its transpose is
    // its inverse.
    factors.u = Mode3TensorMatrixProduct(factors.u, transform.transpose());
    factors.s = Mode3TensorMatrixProduct(factors.s, transform.transpose());
    factors.v = Mode3TensorMatrixProduct(factors.v, transform.transpose());
    return factors;
}

} // namespace tensor_compression
