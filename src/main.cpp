#include <Eigen/SVD>
#include <netcdf>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Test function, see if we can use Eigen to compute the SVD of a 3x3 identity
 * matrix.
 */
static void PrintIdentitySvd() {
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        identity, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d singularValues = svd.singularValues().asDiagonal();
    const Eigen::Matrix3d transposeV = svd.matrixV().transpose();

    if (svd.info() != Eigen::Success ||
        !svd.singularValues().isApprox(Eigen::Vector3d::Ones()) ||
        !(svd.matrixU() * singularValues * transposeV).isApprox(identity)) {
        throw std::runtime_error("Eigen identity SVD verification failed.");
    }
    std::cout << "SVD of the 3x3 identity matrix\n\nU:\n"
              << svd.matrixU() << "\n\nS:\n"
              << singularValues << "\n\nV transpose:\n"
              << transposeV << "\n\n";
}

/**
 * Helper to recursively print the metadata of a NetCDF group and its children.
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

/**
 * Test function, see if we can open netcdf files in the given directory and
 * print their metadata. Tests the netcdf-cxx4 library.
 */
static bool PrintDataMetadata(const std::filesystem::path &directory) {
    if (!std::filesystem::exists(directory)) {
        std::cout << "No data directory at " << directory << ".\n";
        return true;
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
    return success;
}

/**
 * Main entry point for the program.
 *
 * @param argc Number of arguments, including the executable name.
 * @param argv Argument strings.
 *
 * @return true if the program ran successfully, false if there was an error.
 */
int main(int argc, char *argv[]) {
    try {
        if (argc > 2) {
            throw std::invalid_argument(
                "Usage: tensor_compression [data-directory]");
        }
        PrintIdentitySvd();
        const std::filesystem::path directory = argc == 2 ? argv[1] : "data";
        return PrintDataMetadata(directory) ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
