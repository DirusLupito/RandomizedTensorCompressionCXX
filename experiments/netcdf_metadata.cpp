#include "netcdf_metadata.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <netcdf>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensor_compression::experiments {

/**
 * Prints the metadata of a NetCDF group and its children.
 *
 * @param group The group to inspect.
 * @param depth The nesting level used to indent this group.
 * @throws netCDF::exceptions::NcException if its metadata cannot be read.
 */
static void PrintGroupMetadata(const netCDF::NcGroup &group,
                               std::uint32_t depth) {
    const std::string indent(depth * 2, ' ');
    std::cout << indent << "Group: " << group.getName(true) << '\n';
    for (const auto &[name, dimension] : group.getDims()) {
        std::cout << indent << "  Dimension " << name << " = "
                  << dimension.getSize();
        if (dimension.isUnlimited()) {
            std::cout << " (unlimited)";
        }
        std::cout << '\n';
    }
    for (const auto &[name, attribute] : group.getAtts()) {
        std::cout << indent << "  Attribute " << name << ": "
                  << attribute.getType().getName() << "["
                  << attribute.getAttLength() << "]\n";
    }
    for (const auto &[name, variable] : group.getVars()) {
        std::cout << indent << "  Variable " << name << ": "
                  << variable.getType().getName() << " (";
        bool first = true;
        for (const auto &dimension : variable.getDims()) {
            if (!first) {
                std::cout << ", ";
            }
            std::cout << dimension.getName() << "=" << dimension.getSize();
            first = false;
        }
        std::cout << "), " << variable.getAttCount() << " attribute(s)\n";
    }
    for (const auto &[name, child] : group.getGroups()) {
        PrintGroupMetadata(child, depth + 1);
    }
}

void NetcdfMetadataExperiment::RunExperiment() const {
    if (!std::filesystem::exists(directory)) {
        std::cout << "No data directory at " << directory << ".\n";
        return;
    }

    std::vector<std::filesystem::path> files;
    for (const auto &entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".nc") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    std::cout << "Inspecting " << files.size() << " NetCDF file(s) in "
              << directory << " (metadata only).\n";

    bool success = true;
    for (const auto &path : files) {
        std::cout << "\nFile: " << path.filename() << " ("
                  << std::filesystem::file_size(path) << " bytes)\n";
        try {
            const netCDF::NcFile file(path.string(), netCDF::NcFile::read);
            PrintGroupMetadata(file, 0);
        } catch (const netCDF::exceptions::NcException &error) {
            std::cerr << "Cannot inspect " << path << ": " << error.what()
                      << '\n';
            success = false;
        }
    }
    if (!success) {
        throw std::runtime_error(
            "One or more NetCDF files could not be inspected.");
    }
}

} // namespace tensor_compression::experiments
