cmake_minimum_required(VERSION 3.25)
project(RandomizedTensorCompressionCXX VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build configuration" FORCE)
endif()

# Each configuration has its own dependency installation and build tree.
set(CMAKE_CONFIGURATION_TYPES "${CMAKE_BUILD_TYPE}" CACHE STRING "Build configuration" FORCE)
option(RTC_SUPERBUILD "Download and build the numerical and file-format dependencies" ON)
include(cmake/eigen.cmake)

if(RTC_SUPERBUILD)
    include(cmake/dependencies.cmake)
    return()
endif()

find_package(netCDF CONFIG REQUIRED)
find_path(NETCDF_CXX4_INCLUDE_DIR NAMES netcdf REQUIRED)
find_library(NETCDF_CXX4_LIBRARY NAMES netcdf-cxx4 REQUIRED)
add_library(netcdf_cxx4 STATIC IMPORTED)
set_target_properties(netcdf_cxx4 PROPERTIES
    IMPORTED_LOCATION "${NETCDF_CXX4_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NETCDF_CXX4_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES netCDF::netcdf)

add_executable(tensor_compression src/main.cpp)
target_link_libraries(tensor_compression PRIVATE Eigen3::Eigen netcdf_cxx4)
if(NOT RTC_OUTPUT_DIRECTORY)
    set(RTC_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
set_target_properties(tensor_compression PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${RTC_OUTPUT_DIRECTORY}/$<CONFIG>")
if(MSVC)
    target_compile_options(tensor_compression PRIVATE /W4 /permissive- /utf-8)
endif()

# The C library and its HDF5/zlib dependencies are DLLs on Windows. Copy the
# local installation's runtime libraries so the executable runs without PATH edits.
if(WIN32)
    file(GLOB runtime_libraries CONFIGURE_DEPENDS "${RTC_DEPENDENCY_PREFIX}/bin/*.dll")
    add_custom_command(TARGET tensor_compression POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            ${runtime_libraries} "$<TARGET_FILE_DIR:tensor_compression>"
        COMMAND_EXPAND_LISTS
        COMMENT "Copying NetCDF, HDF5, and zlib runtime libraries")
endif()
