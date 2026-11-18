#pragma once

#include "experiment.hpp"

#include <filesystem>
#include <string_view>

namespace tensor_compression::experiments {

/** Runs a DCT timing sweep on the u component of one flow dataset. */
class FlowBenchmark : public Experiment {
public:
    /**
     * Loads the tensor once, constructs the DCT, and records one wall-clock
     * measurement per configuration. Progress is flushed to a
     * partial CSV which is published under its final name on success.
     * @throws std::exception if input, pool creation, decomposition or output
     * fails.
     */
    void RunExperiment() const override;

protected:
    /**
     * @param title The menu label, stored in a string literal.
     * @param stem The NetCDF filename without its extension, also a literal.
     * @param directory The folder containing the NetCDF files.
     * @param batches Whether to run the 32-thread scheduler experiment.
     */
    FlowBenchmark(std::string_view title, std::string_view stem,
                  const std::filesystem::path &directory, bool batches);

private:
    // The data file stem, data directory, and sweep selected by the menu.
    std::string_view stem;
    std::filesystem::path directory;
    bool batches;
};

} // namespace tensor_compression::experiments
