# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CMAKE_CUDA_ARCHITECTURES=native requires >= 3.24
cmake_minimum_required(VERSION 3.24)

set(HOLOSCAN_PACKAGE_NAME holoscan)

# ##############################################################################
# # Options
# ##############################################################################
option(BUILD_SHARED_LIBS "Build Shared Libraries" ON)
option(HOLOSCAN_BUILD_LIBTORCH "Build support for the LibTorch backend" ON)
option(HOLOSCAN_BUILD_ORT "Build support for the ONNX Runtime backend" ON)
option(HOLOSCAN_BUILD_GXF_EXTENSIONS "Build GXF Extensions" ON)
option(HOLOSCAN_BUILD_EXAMPLES "Build Holoscan SDK Examples" ON)
option(HOLOSCAN_BUILD_PYTHON "Build Holoscan SDK Python Bindings" ON)
option(HOLOSCAN_DOWNLOAD_DATASETS "Download SDK Datasets" ON)
option(HOLOSCAN_BUILD_TESTS "Build Holoscan SDK Tests" ON)
option(HOLOSCAN_USE_CCACHE "Use ccache for building Holoscan SDK" OFF)
option(HOLOSCAN_INSTALL_EXAMPLE_SOURCE "Install the example source code" ON)

# ##############################################################################
# # Prerequisite statements
# ##############################################################################

# Set the top-level directory of the project
set(HOLOSCAN_TOP "${CMAKE_CURRENT_LIST_DIR}")

# Append local cmake module path for CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH "${HOLOSCAN_TOP}/cmake/modules")

# Append local cmake dependency path CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${HOLOSCAN_TOP}/cmake/deps")

# Setup rapids-cmake
include(SetupRapidsCMake)

# ##############################################################################
# # Project definition
# ##############################################################################

# Set HOLOSCAN_BUILD_VERSION from 'VERSION' file
unset(HOLOSCAN_BUILD_VERSION CACHE)
file(STRINGS ${HOLOSCAN_TOP}/VERSION HOLOSCAN_BUILD_VERSION)

project(${HOLOSCAN_PACKAGE_NAME}
    VERSION ${HOLOSCAN_BUILD_VERSION}
    DESCRIPTION "Holoscan SDK"
    LANGUAGES C CXX
)
include(SetupCUDA) # CUDA Language enabled there after additional setup

# ##############################################################################
# # Global properties (CMAKE_*)
# ##############################################################################

# Setting the default paths for the build tree
# Libraries (shared and static) are put by default in the lib directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set the default RPATH for built libraries to $ORIGIN and the lib directory
set(CMAKE_BUILD_RPATH "\$ORIGIN:${CMAKE_BINARY_DIR}/lib:${CMAKE_BINARY_DIR}/lib/gxf_extensions")
set(CMAKE_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/gxf_extensions")

# this generates a 'compile_commands.json' file which can be read by VSCode to
# configure the include paths for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Avoid 'Up-to-date' install messages
set(CMAKE_INSTALL_MESSAGE LAZY)

# Default language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Don't append suffix to shared libraries for debug build
set(CMAKE_DEBUG_POSTFIX "")

# ##############################################################################
# # RAPIDS CMAKE configurations
# # (https://docs.rapids.ai/api/rapids-cmake/stable/api.html)
# ##############################################################################

# Set default build type in case not specified
rapids_cmake_build_type(release)

# Generate a C++ header file (under ${CMAKE_CURRENT_BINARY_DIR}) that hold the version (X.Y.Z)
# information of the calling project
rapids_cmake_write_version_file(include/holoscan/version_config.hpp PREFIX HOLOSCAN)

# Get install lib dir
# (https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cmake_install_lib_dir.html)
rapids_cmake_install_lib_dir(HOLOSCAN_INSTALL_LIB_DIR)

# ##############################################################################
# # Setup Cache
# ##############################################################################

set(HOLOSCAN_CACHE_DIR ".cache" CACHE PATH "Directory to use for caching data for CPM and CCache.\
                                            If relative, it is relative to the source directory (CMAKE_SOURCE_DIR).")
mark_as_advanced(HOLOSCAN_CACHE_DIR)

# HOLOSCAN_USE_CCACHE_SKIPPED will be set to TRUE by 'include(SetupCache)'
# CCache is not available.
set(HOLOSCAN_USE_CCACHE_SKIPPED FALSE)
include(SetupCache)

# ##############################################################################
# # Setup hardening security build flags
# ##############################################################################
include(BuildSecurityFlags)

# ##############################################################################
# # Print project settings
# ##############################################################################
include(PrintProjectSettings)

# ##############################################################################
# # Define dependencies
# ##############################################################################

# Include pthread (add 'Threads::Threads' target to link against)
# (https://cmake.org/cmake/help/v3.18/module/FindThreads.html)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

include(cmake/setup_dependencies.cmake)

# ##############################################################################
# # Build sub modules (holoviz, holoinfer)
# ##############################################################################
add_subdirectory(modules)

# ##############################################################################
# # Build core and operators
# ##############################################################################
add_subdirectory(src)

# ##############################################################################
# # Build GXF Extensions and apps
# ##############################################################################
if(HOLOSCAN_BUILD_GXF_EXTENSIONS)
    add_subdirectory(gxf_extensions)
endif()

# ##############################################################################
# # Package project
# ##############################################################################

# strip release binaries
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(_INSTALL_TARGET "install/strip")
else()
    set(_INSTALL_TARGET "install")
endif()

# List targets to export/install
list(APPEND HOLOSCAN_INSTALL_TARGETS
    core
    holoviz
    holoviz_imgui
    holoinfer
    infer_utils
    logger
    op_aja
    op_async_ping_rx
    op_async_ping_tx
    op_bayer_demosaic
    op_format_converter
    op_holoviz
    op_inference
    op_inference_processor
    op_ping_rx
    op_ping_tx
    op_segmentation_postprocessor
    op_video_stream_recorder
    op_video_stream_replayer
    op_v4l2
)

if(HOLOSCAN_BUILD_LIBTORCH)
    list(APPEND HOLOSCAN_INSTALL_TARGETS holoinfer_torch)
endif()
if(HOLOSCAN_BUILD_ORT)
    list(APPEND HOLOSCAN_INSTALL_TARGETS holoinfer_ort)
endif()

# Add PUBLIC dependencies to our install target
# Note: required due to. However, could only export but not install?
# TODO: work on limiting PUBLIC dependencies by not including third-party
# headers in our nterface headers, using forward declaration or PIMPL
list(APPEND HOLOSCAN_INSTALL_TARGETS
    yaml-cpp   # needed by holoscan::core
    glfw       # needed by holoscan::viz
    ajantv2    # needed by holoscan::ops::aja
)

# Copy library files
install(TARGETS ${HOLOSCAN_INSTALL_TARGETS}
    DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}
    EXPORT ${HOLOSCAN_PACKAGE_NAME}-exports
    COMPONENT holoscan-core
)

# Copy headers
install(DIRECTORY include/holoscan/
    DESTINATION include/holoscan
    COMPONENT holoscan-core
)

# Copy bundled tl-expected headers
install(DIRECTORY ${tl-expected_SOURCE_DIR}/include/tl/
    DESTINATION include/3rdparty/tl
    COMPONENT holoscan-dependencies
)

# Copy bundled dlpack headers
install(DIRECTORY ${dlpack_SOURCE_DIR}/include/dlpack/
    DESTINATION include/3rdparty/dlpack
    COMPONENT holoscan-core
)

# Copy bundled cli11 headers
install(DIRECTORY ${cli11_SOURCE_DIR}/include/CLI/
    DESTINATION include/3rdparty/CLI
    COMPONENT holoscan-core
)

# Copy version file
install(FILES ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/include/holoscan/version_config.hpp
    DESTINATION include/holoscan
    COMPONENT holoscan-core
)

# Install GXF
install(DIRECTORY
  ${GXF_core_INCLUDE_DIR}/common
  ${GXF_core_INCLUDE_DIR}/gxf/core
  ${GXF_core_INCLUDE_DIR}/gxf/cuda
  ${GXF_core_INCLUDE_DIR}/gxf/multimedia
  ${GXF_core_INCLUDE_DIR}/gxf/network
  ${GXF_core_INCLUDE_DIR}/gxf/npp
  ${GXF_core_INCLUDE_DIR}/gxf/serialization
  ${GXF_core_INCLUDE_DIR}/gxf/std
DESTINATION "include/gxf"
COMPONENT "holoscan-gxf_libs"
)

# Install CMake script to build GXE applications
install(FILES "${CMAKE_SOURCE_DIR}/cmake/modules/GenerateGXEAppInstall.cmake"
RENAME GenerateGXEApp.cmake
DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
COMPONENT "holoscan-gxf_libs"
)

# Install CMake script for GXF wrapping of extensions
install(FILES "${CMAKE_SOURCE_DIR}/cmake/modules/WrapOperatorAsGXFExtension.cmake"
DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
COMPONENT "holoscan-gxf_libs"
)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/modules/wrap_operator_as_gxf_template"
DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
COMPONENT "holoscan-gxf_libs"
)

# Install CMake scripts to download data from NGC so it can be reused by HoloHub
install(FILES "${CMAKE_SOURCE_DIR}/cmake/modules/HoloscanDownloadData.cmake"
DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
COMPONENT "holoscan-core"
)

install(FILES "${CMAKE_SOURCE_DIR}/scripts/download_ngc_data"
              "${CMAKE_SOURCE_DIR}/scripts/convert_video_to_gxf_entities.py"
              "${CMAKE_SOURCE_DIR}/scripts/gxf_entity_codec.py"
DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
COMPONENT "holoscan-core"
)

# Install CMake script to download example data from NGC
install(FILES "${CMAKE_SOURCE_DIR}/scripts/download_example_data"
DESTINATION "examples"
PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
COMPONENT "holoscan-core"
)

# Define docs and hooks
set(holoscan_doc_string [=[
libholoscan: Holoscan SDK C++ API
]=])

# Defines the install export hook
# We use add_library since we are installing the libraries as part of the SDK
set(holoscan_install_hook_code_string

[=[

if(NOT TARGET fmt::fmt-header-only)
  add_library(fmt::fmt-header-only INTERFACE IMPORTED)
endif()

set(_GXFlibs core std multimedia cuda network npp serialization behavior_tree)

foreach(gxflib IN LISTS _GXFlibs)
  if(NOT TARGET GXF::${gxflib})
    add_library(GXF::${gxflib} SHARED IMPORTED)
    set_target_properties(GXF::${gxflib} PROPERTIES
      IMPORTED_LOCATION "${PACKAGE_PREFIX_DIR}/lib/libgxf_${gxflib}.so"
      IMPORTED_NO_SONAME ON
      INTERFACE_INCLUDE_DIRECTORIES "${PACKAGE_PREFIX_DIR}/include;${PACKAGE_PREFIX_DIR}/include/gxf"
    )
  endif()
endforeach()

if(NOT TARGET GXF::gxe)
  add_executable(GXF::gxe IMPORTED)
  set_target_properties(GXF::gxe PROPERTIES
    IMPORTED_LOCATION "${PACKAGE_PREFIX_DIR}/bin/gxe"
  )
endif()

set(GXF_LIB_DIR "${PACKAGE_PREFIX_DIR}/lib")
set(GXF_EXTENSIONS_DIR "${PACKAGE_PREFIX_DIR}/lib/gxf_extensions")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

]=])

set(holoscan_build_hook_code_string [=[
]=])


# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_export.html#rapids-export
# (rapids_export() always installs to lib and doesn’t use GNUInstallDirs)
rapids_export(
    INSTALL ${HOLOSCAN_PACKAGE_NAME}
    EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    NAMESPACE ${HOLOSCAN_PACKAGE_NAME}::
    VERSION OFF # Forces to not generate the config version file
    DOCUMENTATION holoscan_doc_string
    FINAL_CODE_BLOCK holoscan_install_hook_code_string
)
rapids_export(
    BUILD ${HOLOSCAN_PACKAGE_NAME}
    EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    LANGUAGES C CXX CUDA
    NAMESPACE ${HOLOSCAN_PACKAGE_NAME}::
    DOCUMENTATION holoscan_doc_string
    VERSION OFF # Forces to not generate the config version file
    FINAL_CODE_BLOCK holoscan_build_hook_code_string
)

# Create config version compatible with any newer version of the SDK
# Note: this is to replace the version file from rapids_expert which does not support
#       `AnyNewerVesion`
write_basic_package_version_file(
        "${CMAKE_BINARY_DIR}/${HOLOSCAN_PACKAGE_NAME}-config-version.cmake"
        COMPATIBILITY AnyNewerVersion)

install(FILES "${CMAKE_BINARY_DIR}/${HOLOSCAN_PACKAGE_NAME}-config-version.cmake"
        RENAME "${HOLOSCAN_PACKAGE_NAME}-config-version.cmake"
        DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
        COMPONENT holoscan-core)

# ##############################################################################
# # Download datasets
# ##############################################################################
add_subdirectory(data)

# ##############################################################################
# # Build examples
# ##############################################################################
if(HOLOSCAN_BUILD_EXAMPLES)
  if(HOLOSCAN_BUILD_TESTS)
    include(CTest)
  endif()

  add_subdirectory(examples)
endif()

# ##############################################################################
# # Add tests
# ##############################################################################
if(HOLOSCAN_BUILD_TESTS)
    # Enable testing for the current directory and below
    include(CTest) # it calls 'enable_testing()' internally

    add_subdirectory(tests)

    # add Holoviz tests
    add_test(NAME HOLOVIZ_FUNCTIONAL_TEST COMMAND holoscan::viz::functionaltests)

    # Set Environment variable to resolve memory allocation failure by loading TLS library
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        execute_process(COMMAND nvidia-smi --query-gpu=driver_version --format=csv,noheader OUTPUT_VARIABLE driver_version)
        if(NOT ${driver_version} STREQUAL "")
            string(STRIP ${driver_version} driver_version)
            set(ldpath "LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libnvidia-tls.so.${driver_version}")
            set_property(TEST HOLOVIZ_FUNCTIONAL_TEST PROPERTY ENVIRONMENT ${ldpath})
        else()
            message(WARNING "Could not retrieve driver version with nvidia-smi, can't set LD_PRELOAD for `HOLOVIZ_FUNCTIONAL_TEST`")
        endif()
    endif()

    add_test(NAME HOLOVIZ_UNIT_TEST COMMAND holoscan::viz::unittests)
endif()

if(HOLOSCAN_BUILD_PYTHON)
    add_subdirectory(python)
endif()

include(HoloscanCPack)
