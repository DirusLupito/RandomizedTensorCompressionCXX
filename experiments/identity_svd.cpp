#include "identity_svd.hpp"

#include <Eigen/SVD>
#include <iostream>
#include <stdexcept>

namespace tensor_compression::experiments {

void IdentitySvdExperiment::RunExperiment() const {
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        identity, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d singularValues = svd.singularValues().asDiagonal();
    const Eigen::Matrix3d transposeV = svd.matrixV().transpose();

    if (svd.info() != Eigen::Success ||
        !svd.singularValues().isApprox(Eigen::Vector3d::Ones()) ||
        !(svd.matrixU() * singularValues * transposeV).isApprox(identity)) {
        throw std::runtime_error("Eigen identity SVD verification failed.");
    }
    std::cout << "SVD of the 3x3 identity matrix\n\nU:\n"
              << svd.matrixU() << "\n\nS:\n"
              << singularValues << "\n\nV transpose:\n"
              << transposeV << "\n\n";
}

} // namespace tensor_compression::experiments
