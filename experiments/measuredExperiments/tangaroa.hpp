#pragma once

#include "flow_benchmark.hpp"

namespace tensor_compression::experiments {

/** Measures TSVDM and TSVDM2 on the full Tangaroa velocity field. */
class TangaroaBenchmark final : public FlowBenchmark {
public:
    /**
     * @param directory The folder containing tangaroa.nc.
     * @param batches Whether to measure scheduler batch sizes at 32 threads.
     */
    explicit TangaroaBenchmark(const std::filesystem::path &directory,
                               bool batches = false)
        : FlowBenchmark(batches ? "Tangaroa scheduler batch sizes"
                                : "Tangaroa TSVDM and TSVDM2 timings",
                        "tangaroa", directory, batches) {}
};

} // namespace tensor_compression::experiments
