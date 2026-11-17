#!/usr/bin/env bash
#
# SYNOPSIS
#   Selects a local CMake/generator pair and runs a Linux development command.
#   This is the Linux counterpart of run-cmake.ps1 and mirrors its actions.
#
# USAGE
#   run-cmake.sh [--action ACTION] [--configuration CONFIGURATION] [-- ARGS...]
#
# PARAMETERS
#   --action         configure generates the build; fetch-deps builds
#                    dependencies; build builds the application; run builds and
#                    launches it; nocompile launches an existing executable
#                    without configuring or building; clean removes compiled
#                    build outputs. Defaults to configure.
#   --configuration  RelWithDebInfo (optimized with debug information) or Debug.
#                    Defaults to RelWithDebInfo.
#   ARGS             Arguments forwarded for run or nocompile, including an
#                    optional data directory.
#
# NOTES
#   Build actions require a C/C++ compiler with C++23 support and CMake >= 3.25.
#   Ninja is preferred when installed; Unix Makefiles is the fallback. nocompile
#   only requires an existing executable. Exits non-zero on missing tools,
#   invalid arguments, or failed child processes. Downloads stay in thirdparty;
#   configuration-specific builds and installations stay in build.

set -euo pipefail

# Prints an error and exits, matching the PowerShell driver's throw semantics.
fail() {
    echo "run-cmake.sh: $*" >&2
    exit 1
}

action='configure'
configuration='RelWithDebInfo'
program_arguments=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --action)
            [[ $# -ge 2 ]] || fail '--action requires a value.'
            action="$2"
            shift 2
            ;;
        --configuration)
            [[ $# -ge 2 ]] || fail '--configuration requires a value.'
            configuration="$2"
            shift 2
            ;;
        --)
            shift
            program_arguments=("$@")
            break
            ;;
        *)
            program_arguments+=("$1")
            shift
            ;;
    esac
done

case "$action" in
    configure | fetch-deps | build | run | nocompile | clean) ;;
    *) fail "Invalid action '$action'. Expected configure, fetch-deps, build, run, nocompile, or clean." ;;
esac
case "$configuration" in
    Debug | RelWithDebInfo) ;;
    *) fail "Invalid configuration '$configuration'. Expected Debug or RelWithDebInfo." ;;
esac

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ${#program_arguments[@]} -gt 0 && "$action" != 'run' && "$action" != 'nocompile' ]]; then
    fail 'Program arguments are accepted only with --action run or nocompile.'
fi

# Launch directly from an existing build, preferring the same generator order as
# the build driver. Do not probe tools, configure CMake, or download dependencies.
if [[ "$action" == 'nocompile' ]]; then
    executable=''
    for preset in linux-ninja linux-make; do
        candidate="$repository/build/$preset/$configuration/bin/$configuration/tensor_compression"
        if [[ -f "$candidate" ]]; then
            executable="$candidate"
            break
        fi
    done
    if [[ -z "$executable" ]]; then
        fail "No existing $configuration executable found. Run just build $configuration first."
    fi
    cd "$repository"
    status=0
    "$executable" "${program_arguments[@]}" || status=$?
    if [[ $status -ne 0 ]]; then
        fail "tensor_compression exited with code $status."
    fi
    exit 0
fi

# Find a CMake new enough for the presets file. Prefer Ninja for its parallel,
# incremental builds, and fall back to Unix Makefiles when Ninja is absent.
cmake_executable="$(command -v cmake || true)"
if [[ -z "$cmake_executable" ]]; then
    fail 'Install CMake 3.25 or newer.'
fi
cmake_version="$("$cmake_executable" --version | sed -n 's/^cmake version \([0-9][0-9.]*\).*/\1/p' | head -n 1)"
if [[ -z "$cmake_version" ]] ||
    [[ "$(printf '%s\n' '3.25.0' "$cmake_version" | sort -V | head -n 1)" != '3.25.0' ]]; then
    fail "No compatible CMake >= 3.25 was found (found '${cmake_version:-unknown}')."
fi
if command -v ninja > /dev/null 2>&1; then
    preset='linux-ninja'
elif command -v make > /dev/null 2>&1; then
    preset='linux-make'
else
    fail 'Install Ninja or Make to build with CMake.'
fi

build_directory="$repository/build/$preset/$configuration"
cd "$repository"
if [[ "$action" == 'configure' || ! -f "$build_directory/CMakeCache.txt" ]]; then
    "$cmake_executable" --preset "$preset" -B "$build_directory" "-DCMAKE_BUILD_TYPE=$configuration" ||
        fail 'CMake configuration failed; see the diagnostic above.'
fi
if [[ "$action" == 'configure' ]]; then
    exit 0
fi
if [[ "$action" == 'clean' ]]; then
    # ExternalProject owns independent build systems. Clean the application's
    # actual build tree while preserving the already-built numerical libraries.
    application_build="$build_directory/application"
    if [[ -f "$application_build/CMakeCache.txt" ]]; then
        "$cmake_executable" --build "$application_build" --config "$configuration" --target clean ||
            fail "Cleaning $configuration failed."
    fi
    exit 0
fi
# Bare --parallel lets Make spawn unlimited jobs, so pass the core count to
# match MSBuild's default of one job per logical processor.
build_arguments=(--build "$build_directory" --config "$configuration" --parallel "$(nproc)")
if [[ "$action" == 'fetch-deps' ]]; then
    build_arguments+=(--target dependencies)
fi
"$cmake_executable" "${build_arguments[@]}" ||
    fail "CMake action '$action' failed for $configuration."
if [[ "$action" == 'run' ]]; then
    status=0
    "$build_directory/bin/$configuration/tensor_compression" "${program_arguments[@]}" || status=$?
    if [[ $status -ne 0 ]]; then
        fail "tensor_compression exited with code $status."
    fi
fi
