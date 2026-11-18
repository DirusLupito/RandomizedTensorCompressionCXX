#include "experiment_menu.hpp"

#include "full_tsvdm.hpp"
#include "identity_mode3_product.hpp"
#include "identity_svd.hpp"
#include "identity_tsvdm.hpp"
#include "measuredExperiments/boussinesq.hpp"
#include "measuredExperiments/jungtelziemniak2d.hpp"
#include "measuredExperiments/tangaroa.hpp"
#include "netcdf_metadata.hpp"
#include "parallel_tsvdm.hpp"
#include "reconstructed_tsvdm.hpp"
#include "truncated_tsvdm.hpp"
#include "tsvdm2_energy.hpp"

#include <charconv>
#include <cstddef>
#include <exception>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tensor_compression::experiments {

bool RunExperimentMenu(const std::filesystem::path &directory) {
    // The position in this list gives each experiment its number, starting
    // at one. Append new experiments to keep existing numbers the same.
    const std::unique_ptr<Experiment> experiments[]{
        std::make_unique<IdentitySvdExperiment>(),
        std::make_unique<IdentityMode3ProductExperiment>(),
        std::make_unique<FullTsvdmExperiment>(),
        std::make_unique<TruncatedTsvdmExperiment>(),
        std::make_unique<ReconstructedTsvdmExperiment>(),
        std::make_unique<IdentityTsvdmExperiment>(),
        std::make_unique<ParallelTsvdmExperiment>(),
        std::make_unique<NetcdfMetadataExperiment>(directory),
        std::make_unique<Tsvdm2EnergyExperiment>(),
        std::make_unique<Tsvdm2EnergyExperiment>(2),
        std::make_unique<Jungtelziemniak2dBenchmark>(directory),
        std::make_unique<BoussinesqBenchmark>(directory),
        std::make_unique<TangaroaBenchmark>(directory),
        std::make_unique<Jungtelziemniak2dBenchmark>(directory, true),
        std::make_unique<BoussinesqBenchmark>(directory, true),
        std::make_unique<TangaroaBenchmark>(directory, true),
    };

    bool success = true;
    for (;;) {
        std::cout << "Available experiments\n";
        for (std::size_t i = 0; i < std::size(experiments); ++i) {
            std::cout << i + 1 << ". " << experiments[i]->name << '\n';
        }
        std::cout << "Which experiments? Enter numbers separated by spaces, or "
                     "press Enter to quit.\n> "
                  << std::flush;

        std::string line;
        if (!std::getline(std::cin, line)) {
            if (std::cin.eof() && !std::cin.bad()) {
                return success;
            }
            throw std::runtime_error(
                "Could not read the experiment selection.");
        }

        std::istringstream selection(line);
        std::string token;
        if (!(selection >> token)) {
            return success;
        }

        // Parse one token at a time so an unknown or oversized number cannot
        // prevent the remaining selections on this line from running.
        do {
            std::size_t number = 0;
            const char *end = token.data() + token.size();
            const auto parsed = std::from_chars(token.data(), end, number, 10);
            if (parsed.ec != std::errc{} || parsed.ptr != end || number == 0 ||
                number > std::size(experiments)) {
                continue;
            }
            const Experiment &experiment = *experiments[number - 1];
            std::cout << "Running " << number << ". " << experiment.name
                      << std::endl;
            try {
                experiment.RunExperiment();
            } catch (const std::exception &error) {
                std::cerr << "Experiment " << number
                          << " failed: " << error.what() << '\n';
                success = false;
            }
        } while (selection >> token);
    }
}

} // namespace tensor_compression::experiments
