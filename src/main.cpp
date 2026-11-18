#include "experiment_menu.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>

/**
 * Opens the numbered experiment menu.
 *
 * @param argc The number of arguments, including the executable name.
 * @param argv The executable name and an optional data directory.
 * @return EXIT_SUCCESS if every selected experiment succeeded, or EXIT_FAILURE
 * if a run failed, the arguments were invalid, or the menu could not run.
 */
int main(int argc, char *argv[]) {
    if (argc > 2) {
        std::cerr << "Usage: tensor_compression [data-directory]\n";
        return EXIT_FAILURE;
    }
    try {
        return tensor_compression::experiments::RunExperimentMenu(
                   argc == 2 ? argv[1] : "data")
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
