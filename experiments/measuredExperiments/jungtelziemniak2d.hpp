#pragma once

#include "flow_benchmark.hpp"

namespace tensor_compression::experiments {

/** Measures TSVDM and TSVDM2 on the full Jungtelziemniak2d velocity field. */
class Jungtelziemniak2dBenchmark final : public FlowBenchmark {
public:
    /**
     * @param directory The folder containing jungtelziemniak2d.nc.
     * @param batches Whether to measure scheduler batch sizes at 32 threads.
     */
    explicit Jungtelziemniak2dBenchmark(const std::filesystem::path &directory,
                                        bool batches = false)
        : FlowBenchmark(batches ? "Jungtelziemniak2d scheduler batch sizes"
                                : "Jungtelziemniak2d TSVDM and TSVDM2 timings",
                        "jungtelziemniak2d", directory, batches) {}
};

} // namespace tensor_compression::experiments
