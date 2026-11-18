#pragma once

#include "flow_benchmark.hpp"

namespace tensor_compression::experiments {

/** Measures TSVDM and TSVDM2 on the full Boussinesq velocity field. */
class BoussinesqBenchmark final : public FlowBenchmark {
public:
    /**
     * @param directory The folder containing boussinesq.nc.
     * @param batches Whether to measure scheduler batch sizes at 32 threads.
     */
    explicit BoussinesqBenchmark(const std::filesystem::path &directory,
                                 bool batches = false)
        : FlowBenchmark(batches ? "Boussinesq scheduler batch sizes"
                                : "Boussinesq TSVDM and TSVDM2 timings",
                        "boussinesq", directory, batches) {}
};

} // namespace tensor_compression::experiments
