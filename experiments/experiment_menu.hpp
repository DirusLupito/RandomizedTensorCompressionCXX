#pragma once

#include <filesystem>

namespace tensor_compression::experiments {

/**
 * Lists the experiments and runs each selection in the order entered.
 * A blank line or the end of input closes the menu.
 *
 * @param directory The directory to inspect in the NetCDF experiment.
 * @return true if every selected experiment succeeded, false otherwise.
 * @throws std::runtime_error if the selections cannot be read.
 * @throws std::bad_alloc if there is not enough memory for the menu.
 */
bool RunExperimentMenu(const std::filesystem::path &directory);

} // namespace tensor_compression::experiments
