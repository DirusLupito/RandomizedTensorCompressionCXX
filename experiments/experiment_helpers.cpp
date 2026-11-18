#include "experiment_helpers.hpp"

#include <Eigen/QR>

#include <algorithm>
#include <thread>

namespace tensor_compression::experiments {

Tensor RandomTensor(const std::vector<Eigen::Index> &dimensions,
                    std::mt19937 &generator) {
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    Tensor tensor{dimensions, Eigen::VectorXd(ElementCount(dimensions))};
    for (Eigen::Index i = 0; i < tensor.values.size(); ++i) {
        tensor.values(i) = distribution(generator);
    }
    return tensor;
}

Eigen::MatrixXd RandomOrthogonal(Eigen::Index size, std::mt19937 &generator) {
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    Eigen::MatrixXd random(size, size);
    for (Eigen::Index i = 0; i < random.size(); ++i) {
        random(i) = distribution(generator);
    }
    return Eigen::HouseholderQR<Eigen::MatrixXd>(random).householderQ();
}

bool SameTensor(const Tensor &left, const Tensor &right) {
    return left.dimensions == right.dimensions &&
           left.values.size() == right.values.size() &&
           left.values == right.values;
}

AsyncParallelForDriver &GetExperimentDriver() {
    static AsyncParallelForDriver driver(
        std::max(1U, std::thread::hardware_concurrency()));
    return driver;
}

} // namespace tensor_compression::experiments
