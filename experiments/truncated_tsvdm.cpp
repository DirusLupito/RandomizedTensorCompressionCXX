#include "truncated_tsvdm.hpp"

#include "experiment_helpers.hpp"
#include "truncate_tsvdm.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tensor_compression::experiments {

/**
 * Checks that truncated star-M SVD factors have MATLAB's zero-padded shapes,
 * U m x r x n, S r x r x n, and V p x r x n, where r is the largest rank.
 *
 * @param factors The truncated factors to check.
 * @param m The number of rows in each slice of the original tensor.
 * @param p The number of columns in each slice of the original tensor.
 * @param rank The largest rank kept in any slice, r.
 * @param n The number of frontal slices.
 *
 * @throws std::runtime_error if any factor has other dimensions.
 */
static void CheckTruncatedShapes(const TsvdmFactors &factors, Eigen::Index m,
                                 Eigen::Index p, Eigen::Index rank,
                                 Eigen::Index n) {
    if (factors.u.dimensions != std::vector<Eigen::Index>{m, rank, n} ||
        factors.s.dimensions != std::vector<Eigen::Index>{rank, rank, n} ||
        factors.v.dimensions != std::vector<Eigen::Index>{p, rank, n}) {
        throw std::runtime_error(
            "Truncated star-M SVD factors have the wrong dimensions.");
    }
}

void TruncatedTsvdmExperiment::RunExperiment() const {
    AsyncParallelForDriver &driver = GetExperimentDriver();

    // The same tall and wide shapes as the full SVD experiment, so that reading
    // m and p from the wrong factor would be caught for economy factors.
    const std::array<std::array<Eigen::Index, 3>, 2> shapes{
        {{6, 4, 5}, {3, 5, 4}}};
    std::mt19937 generator(549);
    for (const auto &[m, p, n] : shapes) {
        const Tensor tensor = RandomTensor({m, p, n}, generator);
        const Eigen::MatrixXd transform = RandomOrthogonal(n, generator);

        // Uniform ranks of 0, 2, and min(m, p) cover an empty result, a typical
        // truncation, and no truncation at all. The mixed ranks cycle through
        // 0 to min(m, p), so one list holds both extremes, and the slices with
        // smaller ranks need padding.
        const Eigen::Index q = std::min(m, p);
        const std::size_t count = static_cast<std::size_t>(n);
        std::vector<Eigen::Index> mixed(count);
        for (std::size_t i = 0; i < count; ++i) {
            mixed[i] = static_cast<Eigen::Index>(i) % (q + 1);
        }
        const std::array<std::vector<Eigen::Index>, 4> rankLists{
            std::vector<Eigen::Index>(count, 0),
            std::vector<Eigen::Index>(count, 2),
            std::vector<Eigen::Index>(count, q), mixed};

        for (const bool economy : {false, true}) {
            const TsvdmFactors untruncated =
                Tsvdm(tensor, transform, driver, {}, true, economy);
            for (const std::vector<Eigen::Index> &ranks : rankLists) {
                const TsvdmFactors truncated =
                    TruncateTsvdm(untruncated, ranks);
                const Eigen::Index r = std::ranges::max(ranks);
                CheckTruncatedShapes(truncated, m, p, r, n);

                // Truncation only copies entries, so each slice must match
                // its leading triples exactly, followed by zeros.
                for (Eigen::Index i = 0; i < n; ++i) {
                    const Eigen::Index k = ranks[static_cast<std::size_t>(i)];
                    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(m, r);
                    u.leftCols(k) = FrontalSlice(untruncated.u, i).leftCols(k);
                    Eigen::MatrixXd s = Eigen::MatrixXd::Zero(r, r);
                    s.topLeftCorner(k, k) =
                        FrontalSlice(untruncated.s, i).topLeftCorner(k, k);
                    Eigen::MatrixXd v = Eigen::MatrixXd::Zero(p, r);
                    v.leftCols(k) = FrontalSlice(untruncated.v, i).leftCols(k);
                    if (FrontalSlice(truncated.u, i) != u ||
                        FrontalSlice(truncated.s, i) != s ||
                        FrontalSlice(truncated.v, i) != v) {
                        throw std::runtime_error(
                            "Truncated star-M SVD slice does not hold exactly "
                            "the leading singular triples.");
                    }
                }

                // Tsvdm runs the same SVDs and the same copies, so its
                // truncation must agree bit for bit. In the original domain,
                // this check compares the shapes.
                const TsvdmFactors viaTsvdm =
                    Tsvdm(tensor, transform, driver, ranks, true, economy);
                if (!SameTensor(viaTsvdm.u, truncated.u) ||
                    !SameTensor(viaTsvdm.s, truncated.s) ||
                    !SameTensor(viaTsvdm.v, truncated.v)) {
                    throw std::runtime_error(
                        "Tsvdm truncation disagrees with TruncateTsvdm.");
                }
                CheckTruncatedShapes(
                    Tsvdm(tensor, transform, driver, ranks, false, economy), m,
                    p, r, n);
            }
            std::cout << "Truncated star-M SVD of a random " << m << "x" << p
                      << "x" << n << " tensor ("
                      << (economy ? "economy" : "full") << "): uniform 0, 2, "
                      << q << ", and mixed ranks match\n";
        }

        // Too few ranks, a negative rank, and a rank above min(m, p) must all
        // be rejected, both directly and through Tsvdm.
        std::vector<Eigen::Index> tooFew(count - 1, 1);
        std::vector<Eigen::Index> negative(count, 1);
        negative.front() = -1;
        std::vector<Eigen::Index> tooLarge(count, 1);
        tooLarge.back() = q + 1;
        const TsvdmFactors untruncated =
            Tsvdm(tensor, transform, driver, {}, true, false);
        for (const std::vector<Eigen::Index> *ranks :
             {&tooFew, &negative, &tooLarge}) {
            bool truncateRejected = false;
            try {
                TruncateTsvdm(untruncated, *ranks);
            } catch (const std::invalid_argument &) {
                truncateRejected = true;
            }
            bool tsvdmRejected = false;
            try {
                Tsvdm(tensor, transform, driver, *ranks, true, false);
            } catch (const std::invalid_argument &) {
                tsvdmRejected = true;
            }
            if (!truncateRejected || !tsvdmRejected) {
                throw std::runtime_error(
                    "Invalid truncation ranks were accepted.");
            }
        }
    }
    std::cout << "Truncated star-M SVD verification passed.\n\n";
}

} // namespace tensor_compression::experiments
