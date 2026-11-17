set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# List available commands.
default:
    @just --list

# Configure the native x64 build and download Eigen.
[windows]
configure configuration="RelWithDebInfo":
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/run-cmake.ps1 -Action configure -Configuration '{{configuration}}'

# Download and build the pinned NetCDF/HDF5/zlib dependencies.
[windows]
fetch-deps configuration="RelWithDebInfo":
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/run-cmake.ps1 -Action fetch-deps -Configuration '{{configuration}}'

# Build incrementally. Optimized builds include PDB debug information.
[windows]
build configuration="RelWithDebInfo":
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/run-cmake.ps1 -Action build -Configuration '{{configuration}}'

# Build the optimized executable with debug information.
[windows]
[linux]
release: (build "RelWithDebInfo")

# Build and run RelWithDebInfo. Optionally pass a data-directory path.
[windows]
[script("powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File")]
[positional-arguments]
[extension(".ps1")]
run *args:
    $ErrorActionPreference = 'Stop'
    & ./scripts/run-cmake.ps1 -Action run -Configuration RelWithDebInfo -ProgramArguments $args

# Run the existing optimized executable with debug information without building.
[windows]
[script("powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File")]
[positional-arguments]
[extension(".ps1")]
nocompile *args:
    $ErrorActionPreference = 'Stop'
    & ./scripts/run-cmake.ps1 -Action nocompile -Configuration RelWithDebInfo -ProgramArguments $args

# Build and run without optimization. Optionally pass a data-directory path.
[windows]
[script("powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File")]
[positional-arguments]
[extension(".ps1")]
debug *args:
    $ErrorActionPreference = 'Stop'
    & ./scripts/run-cmake.ps1 -Action run -Configuration Debug -ProgramArguments $args

# Clean the application while retaining built dependencies and configuration.
[windows]
clean configuration="RelWithDebInfo":
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/run-cmake.ps1 -Action clean -Configuration '{{configuration}}'

# Configure the native build and download Eigen.
[linux]
configure configuration="RelWithDebInfo":
    ./scripts/run-cmake.sh --action configure --configuration '{{configuration}}'

# Download and build the pinned NetCDF/HDF5/zlib dependencies.
[linux]
fetch-deps configuration="RelWithDebInfo":
    ./scripts/run-cmake.sh --action fetch-deps --configuration '{{configuration}}'

# Build incrementally. Optimized builds include debug information.
[linux]
build configuration="RelWithDebInfo":
    ./scripts/run-cmake.sh --action build --configuration '{{configuration}}'

# Build and run RelWithDebInfo. Optionally pass a data-directory path.
[linux]
[positional-arguments]
run *args:
    ./scripts/run-cmake.sh --action run --configuration RelWithDebInfo -- "$@"

# Run the existing optimized executable with debug information without building.
[linux]
[positional-arguments]
nocompile *args:
    ./scripts/run-cmake.sh --action nocompile --configuration RelWithDebInfo -- "$@"

# Build and run without optimization. Optionally pass a data-directory path.
[linux]
[positional-arguments]
debug *args:
    ./scripts/run-cmake.sh --action run --configuration Debug -- "$@"

# Clean the application while retaining built dependencies and configuration.
[linux]
clean configuration="RelWithDebInfo":
    ./scripts/run-cmake.sh --action clean --configuration '{{configuration}}'
