#pragma once

#include <string_view>

namespace tensor_compression::experiments {

/**
 * Represents an experiment that can be selected from the numbered menu.
 */
class Experiment {
public:
    // The name printed in the menu and before each run.
    const std::string_view name;

    /** Releases resources owned by the selected experiment. */
    virtual ~Experiment() = default;

    /**
     * Runs the experiment once and prints its results.
     * Exceptions from a failed run are reported by the menu.
     */
    virtual void RunExperiment() const = 0;

protected:
    /**
     * @param displayName The name to print. Its storage must outlive the
     * experiment.
     */
    explicit Experiment(std::string_view displayName) : name(displayName) {}
};

} // namespace tensor_compression::experiments
