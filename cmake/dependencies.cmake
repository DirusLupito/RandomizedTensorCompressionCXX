include(ExternalProject)

# NetCDF's upstream projects expect installed dependencies during configuration.
# A standard superbuild preserves those build systems without patching sources or
# supplying replacement CMakeLists.txt files. Installations stay local to the build.
set(dependency_prefix "${CMAKE_BINARY_DIR}/dependencies")
set(dependency_arguments
    "-DCMAKE_INSTALL_PREFIX:PATH=${dependency_prefix}"
    "-DCMAKE_PREFIX_PATH:PATH=${dependency_prefix}"
    "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_CONFIGURATION_TYPES:STRING=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_INSTALL_LIBDIR:STRING=lib"
    "-DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5"
    -DBUILD_TESTING:BOOL=OFF)
if(MSVC)
    # Parallel compilation matters for the large HDF5 and netCDF source trees.
    # INIT preserves CMake's normal platform flags when a build is first configured.
    list(APPEND dependency_arguments
        -DCMAKE_C_FLAGS_INIT:STRING=/MP
        -DCMAKE_CXX_FLAGS_INIT:STRING=/MP)
endif()

ExternalProject_Add(zlib
    URL https://github.com/madler/zlib/archive/refs/tags/v1.3.1.tar.gz
    URL_HASH SHA256=17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c
    DOWNLOAD_NAME zlib-1.3.1.tar.gz
    DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/thirdparty/downloads"
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/zlib-1.3.1"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_ARGS ${dependency_arguments} -DZLIB_BUILD_EXAMPLES:BOOL=OFF)

ExternalProject_Add(hdf5
    DEPENDS zlib
    URL https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_1.14.6.tar.gz
    URL_HASH SHA256=09ee1c671a87401a5201c06106650f62badeea5a3b3941e9b1e2e1e08317357f
    DOWNLOAD_NAME hdf5-1.14.6.tar.gz
    DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/thirdparty/downloads"
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/hdf5-1.14.6"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_ARGS ${dependency_arguments}
        -DBUILD_SHARED_LIBS:BOOL=ON
        -DBUILD_STATIC_LIBS:BOOL=OFF
        -DHDF5_BUILD_CPP_LIB:BOOL=OFF
        -DHDF5_BUILD_FORTRAN:BOOL=OFF
        -DHDF5_BUILD_JAVA:BOOL=OFF
        -DHDF5_BUILD_HL_LIB:BOOL=ON
        -DHDF5_BUILD_TOOLS:BOOL=OFF
        -DHDF5_BUILD_UTILS:BOOL=OFF
        -DHDF5_BUILD_EXAMPLES:BOOL=OFF
        -DHDF5_ENABLE_SZIP_SUPPORT:BOOL=OFF
        -DHDF5_ENABLE_Z_LIB_SUPPORT:BOOL=ON
        -DHDF5_ENABLE_NONSTANDARD_FEATURE_FLOAT16:BOOL=OFF)

ExternalProject_Add(netcdf_c
    DEPENDS hdf5
    URL https://downloads.unidata.ucar.edu/netcdf-c/4.10.1/netcdf-c-4.10.1.tar.gz
    URL_HASH SHA256=db3b69ff4a5ee1a7d79a5c36664d2128b752c266e966369fcf7311ec5f927564
    DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/thirdparty/downloads"
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/netcdf-c-4.10.1"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_ARGS ${dependency_arguments}
        -DBUILD_SHARED_LIBS:BOOL=ON
        -DNETCDF_ENABLE_HDF5:BOOL=ON
        -DNETCDF_ENABLE_DAP:BOOL=OFF
        -DNETCDF_ENABLE_BYTERANGE:BOOL=OFF
        -DNETCDF_ENABLE_REMOTE_FUNCTIONALITY:BOOL=OFF
        -DNETCDF_ENABLE_NCZARR:BOOL=OFF
        -DNETCDF_ENABLE_PLUGINS:BOOL=OFF
        -DNETCDF_ENABLE_LIBXML2:BOOL=OFF
        -DNETCDF_ENABLE_TESTS:BOOL=OFF
        -DNETCDF_ENABLE_EXAMPLES:BOOL=OFF
        -DNETCDF_BUILD_UTILITIES:BOOL=OFF
        -DNETCDF_PACKAGE:BOOL=OFF)

ExternalProject_Add(netcdf_cxx4
    DEPENDS netcdf_c
    URL https://downloads.unidata.ucar.edu/netcdf-cxx/4.3.1/netcdf-cxx4-4.3.1.tar.gz
    URL_HASH SHA256=6a1189a181eed043b5859e15d5c080c30d0e107406fbb212c8fb9814e90f3445
    DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/thirdparty/downloads"
    SOURCE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/netcdf-cxx4-4.3.1"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CMAKE_ARGS ${dependency_arguments}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        # The C++ release expects FindHDF5's legacy variable even when it loads
        # HDF5's modern config package. Supply the installed shared-library target.
        -DHDF5_C_LIBRARY_hdf5:STRING=hdf5-shared
        -DNCXX_ENABLE_TESTS:BOOL=OFF
        -DENABLE_DOXYGEN:BOOL=OFF)

add_custom_target(dependencies DEPENDS netcdf_cxx4)

# Configure the application only after the upstream libraries are installed.
# BUILD_ALWAYS runs its incremental build so edits to our own source are picked up.
ExternalProject_Add(application
    DEPENDS netcdf_cxx4
    SOURCE_DIR "${PROJECT_SOURCE_DIR}"
    BINARY_DIR "${CMAKE_BINARY_DIR}/application"
    DOWNLOAD_COMMAND ""
    CMAKE_ARGS ${dependency_arguments}
        -DRTC_SUPERBUILD:BOOL=OFF
        "-DRTC_DEPENDENCY_PREFIX:PATH=${dependency_prefix}"
        "-DRTC_OUTPUT_DIRECTORY:PATH=${CMAKE_BINARY_DIR}/bin"
    BUILD_ALWAYS TRUE
    INSTALL_COMMAND "")
