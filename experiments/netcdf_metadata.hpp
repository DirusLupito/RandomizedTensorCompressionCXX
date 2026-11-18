#pragma once

#include "experiment.hpp"

#include <filesystem>
#include <utility>

namespace tensor_compression::experiments {

/** Lists the contents of the selected NetCDF files without loading their data.
 */
class NetcdfMetadataExperiment final : public Experiment {
public:
    /**
     * @param dataDirectory The directory containing the NetCDF files to
     * inspect.
     */
    explicit NetcdfMetadataExperiment(std::filesystem::path dataDirectory)
        : Experiment("NetCDF metadata"), directory(std::move(dataDirectory)) {}

    /**
     * Prints the dimensions, attributes, and variables of each NetCDF file.
     *
     * @throws std::filesystem::filesystem_error if the directory cannot be
     * read.
     * @throws std::runtime_error if any NetCDF file cannot be inspected.
     */
    void RunExperiment() const override;

private:
    // The directory selected on the command line, or the default data
    // directory.
    std::filesystem::path directory;
};

} // namespace tensor_compression::experiments
