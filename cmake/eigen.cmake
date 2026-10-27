include(FetchContent)

# Keep downloaded sources outside configuration-specific build trees. Eigen is
# header-only; its tests, examples, and documentation are unnecessary here.
set(BUILD_TESTING OFF CACHE BOOL "Build dependency tests" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Build Eigen documentation" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Generate Eigen pkg-config files" FORCE)
# Eigen 3.4 probes Fortran for its optional LAPACK target even with tests off.
# This project only uses C++ headers; avoid launching Visual Studio's Fortran probe.
set(CMAKE_Fortran_COMPILER NOTFOUND CACHE FILEPATH "Fortran is not used by this project" FORCE)
FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/thirdparty/downloads"
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/eigen-3.4.0"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(eigen)
