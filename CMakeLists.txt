# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# RMM dependency requires >= 3.26.4
cmake_minimum_required(VERSION 3.26.4)

set(HOLOSCAN_PACKAGE_NAME holoscan)

# ##############################################################################
# # Options
# ##############################################################################
option(BUILD_SHARED_LIBS "Build Shared Libraries" ON)
option(HOLOSCAN_BUILD_HOLOLINK "Extend Holoscan SDK build with Holoscan Sensor Bridge targets" OFF)
option(HOLOSCAN_BUILD_LIBTORCH "Build support for the LibTorch backend" ON)
option(HOLOSCAN_BUILD_ORT "Build support for the ONNX Runtime backend" ON)
option(HOLOSCAN_BUILD_EXAMPLES "Build Holoscan SDK Examples" ON)
option(HOLOSCAN_BUILD_PYTHON "Build Holoscan SDK Python Bindings" ON)
option(HOLOSCAN_DOWNLOAD_DATASETS "Download SDK Datasets" ON)
option(HOLOSCAN_BUILD_TESTS "Build Holoscan SDK Tests" ON)
option(HOLOSCAN_USE_SCCACHE "Use sccache for building Holoscan SDK" ON)
option(HOLOSCAN_INSTALL_EXAMPLE_SOURCE "Install the example source code" ON)
option(HOLOSCAN_ENABLE_CLANG_TIDY "Enable use of clang-tidy" OFF)
option(HOLOSCAN_CLANG_TIDY_CHECK_HEADERS "Enable clang-tidy header checking (requires HOLOSCAN_ENABLE_CLANG_TIDY)" OFF)
option(HOLOSCAN_ENABLE_GOOGLE_SANITIZER "Enable use of google sanitizer" OFF)
option(HOLOSCAN_BUILD_GXF_EXTENSIONS "Build GXF Extensions" ON)
option(HOLOSCAN_BUILD_IPC
  "Build the holo_ipc module (holoscan::ipc). When OFF, the module is omitted entirely. When ON, the library is built; see HOLOSCAN_IPC_TRANSPORT_FASTDDS for Fast DDS."
  OFF
)
option(HOLOSCAN_IPC_TRANSPORT_FASTDDS
  "When HOLOSCAN_BUILD_IPC is ON, add the Fast DDS transport (and Fast DDS examples). When OFF, holo_ipc is core-only; no other DDS transport is built in-tree."
  ON
)
option(HOLOSCAN_ALLOW_SYSTEM_INSTALL "Allow installation to system directories" OFF)

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

# If enabling google sanitizer
if(HOLOSCAN_ENABLE_GOOGLE_SANITIZER)
  # Note: Before running the tests, the following command must be executed:
  #   export ASAN_OPTIONS=symbolize=1:protect_shadow_gap=0
  # This ensures that stack traces are symbolized and prevents false positives related to shadow gap
  # protection.
  # Without this, CUDA-related tests may fail (e.g., CUDA runtime API error "out of memory").
  # (See https://github.com/NVIDIA/DALI/pull/362).
  # You may still see some false positives when running the tests that uses the Vulkan-related APIs.
  list(APPEND CMAKE_CXX_FLAGS "-g -O1 -Wno-stringop-truncation -fsanitize=address -fno-omit-frame-pointer")
endif()

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

# Create a symbolic link for compile_commands.json in the source tree when building out of source
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(STATUS "Skipping compile_commands.json symlink because this is an in-source build.")
else()
    # Clangd searches parent directories and 'build/' folders for compile_commands.json.
    # To ensure Clangd can find it, we create a symlink in the source directory.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${CMAKE_BINARY_DIR}/compile_commands.json
                ${CMAKE_SOURCE_DIR}/compile_commands.json
        RESULT_VARIABLE result
        ERROR_VARIABLE error
    )
    if(result)
        message(FATAL_ERROR "Failed to create symbolic link for compile_commands.json: ${error}")
    endif()
endif()

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

option(HOLOSCAN_USE_CACHE_DIR "Use CPM and compiler cache directory (set OFF for e.g. OpenEmbedded)" ON)
set(HOLOSCAN_CACHE_DIR ".cache" CACHE PATH "Directory for CPM and compiler caches (used when HOLOSCAN_USE_CACHE_DIR is ON).\
                                            If relative, it is relative to the source directory (CMAKE_SOURCE_DIR).")
mark_as_advanced(HOLOSCAN_CACHE_DIR)

# HOLOSCAN_USE_SCCACHE_SKIPPED will be set to TRUE by 'include(SetupCache)'
# if SCCache is not available.
set(HOLOSCAN_USE_SCCACHE_SKIPPED FALSE)
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

# If enabling clang-tidy
if(HOLOSCAN_ENABLE_CLANG_TIDY)
  # Common clang-tidy checks: enable all ("*"), then disable noisy or low-signal groups for this
  # codebase. Joined into the comma-separated form required by clang-tidy --checks=...
  # Wildcards (e.g. -cppcoreguidelines-*, -readability-*) subsume same-prefix checks; do not list
  # those checks again. Disable list is sorted alphabetically (after "*").
  set(_HOLOSCAN_CLANG_TIDY_CHECK_LIST
      "*"
      "-abseil-*"
      "-altera-*"
      "-android-*"
      "-bugprone-assignment-in-if-condition"
      "-bugprone-branch-clone"
      "-bugprone-easily-swappable-parameters"
      "-bugprone-empty-catch"
      "-bugprone-exception-escape"
      "-bugprone-implicit-widening-of-multiplication-result"
      "-bugprone-incorrect-roundings"
      "-bugprone-macro-parentheses"
      "-bugprone-multi-level-implicit-pointer-conversion"
      "-bugprone-narrowing-conversions"
      "-bugprone-switch-missing-default-case"
      "-bugprone-inc-dec-in-conditions"
      "-bugprone-casting-through-void"
      "-cert-*"
      "-clang-diagnostic-ignored-optimization-argument"
      "-clang-diagnostic-potentially-evaluated-expression"
      "-concurrency-mt-unsafe"
      "-cppcoreguidelines-*"
      "-fuchsia-*"
      "-google-*"
      "-hicpp-*"
      "-llvm-else-after-return"
      "-llvm-header-guard"
      "-llvm-include-order"
      "-llvm-qualified-auto"
      "-llvmlibc-callee-namespace"
      "-llvmlibc-implementation-in-namespace"
      "-llvmlibc-inline-function-decl"
      "-llvmlibc-restrict-system-libc-headers"
      "-misc-*"
      "-modernize-*"
      "-performance-enum-size"
      "-performance-inefficient-string-concatenation"
      "-performance-move-const-arg"
      "-performance-no-automatic-move"
      "-performance-no-int-to-ptr"
      "-performance-unnecessary-value-param"
      "-portability-simd-intrinsics"
      "-readability-*")
  string(JOIN "," CLANG_TIDY_CHECKS ${_HOLOSCAN_CLANG_TIDY_CHECK_LIST})
  unset(_HOLOSCAN_CLANG_TIDY_CHECK_LIST)

  if(HOLOSCAN_CLANG_TIDY_CHECK_HEADERS)
    # Header filter to check project headers while excluding third-party code.
    # Note: --exclude-header-filter is not available until clang-tidy>=19.0, so we use
    # a positive regex that explicitly matches the directories we want to check.
    # For holoviz, we match src/tests/examples subdirectories to exclude thirdparty/.
    set(CLANG_TIDY_HEADER_FILTER
        ".*/include/holoscan/.*|.*/python/holoscan/.*|.*/gxf_extensions/.*|.*/modules/holoinfer/.*|.*/modules/holoviz/(src|tests|examples)/.*"
    )
    set(CMAKE_CXX_CLANG_TIDY
        clang-tidy
        "--header-filter=${CLANG_TIDY_HEADER_FILTER}"
        "--checks=${CLANG_TIDY_CHECKS}"
        "--extra-arg=-std=c++17"
    )
  else()
    # Without header filter, only check source files (not included headers)
    set(CMAKE_CXX_CLANG_TIDY
        clang-tidy
        "--checks=${CLANG_TIDY_CHECKS}"
        "--extra-arg=-std=c++17"
    )
  endif()
endif()

# Suppress the warning: "warning #128: type qualifiers ignored on cast result type"
add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=128>
)

# ##############################################################################
# # Build sub modules (holoviz, holoinfer, holoipc)
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
# # Build Python module
# ##############################################################################

# Pybind11 ABI config
#
# Prevent strict ABI protection in pybind11
# https://github.com/pybind/pybind11/pull/2602
#
# Intentionally defined even in C++-only builds that may be mixed with Python
# bindings at a later time, such as in the case of Debian development packages
# and non-development Python wheels.
add_library(pybind11_abi_config INTERFACE)
target_compile_definitions(pybind11_abi_config
    INTERFACE
        PYBIND11_COMPILER_TYPE=""
        PYBIND11_STDLIB=""
        PYBIND11_BUILD_ABI=""
)

# Use it as holoscan::pybind11 in this build and projects importing holoscan
add_library(holoscan::pybind11 ALIAS pybind11_abi_config)
set_target_properties(pybind11_abi_config PROPERTIES
    EXPORT_NAME pybind11
)

if(HOLOSCAN_BUILD_PYTHON)
    if(HOLOSCAN_BUILD_TESTS)
       include(CTest)
    endif()
    add_subdirectory(python)
endif()

# ##############################################################################
# # Configure scripts
# ##############################################################################
add_subdirectory(scripts)

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
    add_test(NAME HOLOVIZ_UNIT_TEST COMMAND holoscan::viz::unittests)
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
    cuda_helper
    gpu_resident_cuda
    holoviz
    holoviz_imgui
    holoinfer
    holoinfer_utils
    infer_utils
    logger
    matx
    data_loggers_async_console_logger
    data_loggers_basic_console_logger
    op_async_ping_rx
    op_async_ping_tx
    op_bayer_demosaic
    op_bayer_demosaic_gpu_resident
    op_format_converter
    op_format_converter_gpu_resident
    op_gxf_codelet
    op_holoviz
    op_raw_image_processor
    op_inference
    op_inference_processor
    op_gpu_resident_inference
    op_ping_rx
    op_ping_tx
    op_ping_tensor_tx
    op_ping_tensor_rx
    op_segmentation_postprocessor
    op_test_ops
    op_video_stream_recorder
    op_video_stream_replayer
    op_v4l2
    op_video_io
    pose_tree
    pubsub_common
    pubsub_in_memory
    pubsub_runtime
    profiler
    spdlog_logger
)
if(HOLOSCAN_BUILD_LIBTORCH)
  list(APPEND HOLOSCAN_INSTALL_TARGETS holoinfer_torch)
endif()
if(HOLOSCAN_BUILD_ORT)
    list(APPEND HOLOSCAN_INSTALL_TARGETS holoinfer_ort)
endif()
if(HOLOSCAN_BUILD_IPC)
    list(APPEND HOLOSCAN_INSTALL_TARGETS holo_ipc)
    if(HOLOSCAN_IPC_TRANSPORT_FASTDDS)
        list(APPEND HOLOSCAN_INSTALL_TARGETS ipc_fastdds_deps pubsub_fastdds)
    endif()
endif()
list(APPEND HOLOSCAN_INSTALL_TARGETS pybind11_abi_config)

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

# Copy version file
install(FILES ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/include/holoscan/version_config.hpp
    DESTINATION include/holoscan
    COMPONENT holoscan-core
)

# Apache 2.0 license at install prefix root
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE.txt"
    DESTINATION .
    COMPONENT holoscan-core
)

# Install GXF
install(DIRECTORY
  ${GXF_INCLUDE_DIR}/common
  ${GXF_INCLUDE_DIR}/gxf/app
  ${GXF_INCLUDE_DIR}/gxf/core
  ${GXF_INCLUDE_DIR}/gxf/cuda
  ${GXF_INCLUDE_DIR}/gxf/logger
  ${GXF_INCLUDE_DIR}/gxf/multimedia
  ${GXF_INCLUDE_DIR}/gxf/pubsub
  ${GXF_INCLUDE_DIR}/gxf/rmm
  ${GXF_INCLUDE_DIR}/gxf/serialization
  ${GXF_INCLUDE_DIR}/gxf/std
  ${GXF_INCLUDE_DIR}/gxf/ucx
DESTINATION "include/gxf"
COMPONENT "holoscan-gxf_libs"
)

set(HOLOSCAN_GXF_INSTALL_HOOK [=[
if(NOT TARGET GXF::common)
  add_library(GXF::common INTERFACE IMPORTED)
  set_target_properties(GXF::common PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PACKAGE_PREFIX_DIR}/include/gxf")
endif()

]=])
foreach(_component ${HOLOSCAN_GXF_COMPONENTS})
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type)
    get_target_property(GXF_${_component}_LOCATION GXF::${_component} IMPORTED_LOCATION_${_build_type})
    if(NOT GXF_${_component}_LOCATION)
        get_target_property(GXF_${_component}_LOCATION GXF::${_component} IMPORTED_LOCATION)
    endif()
        install(FILES "${GXF_${_component}_LOCATION}"
            DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}
            COMPONENT "holoscan-gxf_libs"
        )

        get_target_property(interface_targets GXF::${_component} INTERFACE_LINK_LIBRARIES)
        if(interface_targets)
            # GXF "gems" are a subcomponent of the GXF::std extension and are not distributed individually
            list(FILTER interface_targets EXCLUDE REGEX ".*gems.*")
            # "internal" targets are a build detail and are no-ops for public interface
            list(FILTER interface_targets EXCLUDE REGEX ".*_internal.*")
            # RMM headers are embedded in the Holoscan SDK package
            #list(FILTER interface_targets EXCLUDE REGEX "^rmm::rmm$")
            set(HOLOSCAN_GXF_INSTALL_HOOK "${HOLOSCAN_GXF_INSTALL_HOOK}
if(TARGET GXF::${_component})
set_target_properties(GXF::${_component} PROPERTIES \
  INTERFACE_LINK_LIBRARIES \"${interface_targets}\"\
)
endif()"
            )
        endif()
endforeach()

# Install CMake scripts to download data from NGC so it can be reused by HoloHub
install(FILES "${CMAKE_SOURCE_DIR}/cmake/modules/HoloscanDownloadData.cmake"
DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
COMPONENT "holoscan-core"
)
# FindTinyXML2.cmake: used by holoscan-config.cmake when TinyXML2Config.cmake is absent
# (e.g. Debian libtinyxml2-dev). Must live next to holoscan-config.cmake on CMAKE_MODULE_PATH.
install(FILES "${CMAKE_SOURCE_DIR}/cmake/modules/FindTinyXML2.cmake"
        DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
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
include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit)
find_dependency(CCCL 3.2 PATHS ${CMAKE_CURRENT_LIST_DIR}/../cccl /usr/local/cuda/lib64/cmake/cccl)

find_dependency(yaml-cpp HINTS ${CMAKE_CURRENT_LIST_DIR}/../yaml-cpp)
# Legacy non-namespaced yaml-cpp linkage required for GXF internal targets
if(TARGET yaml-cpp::yaml-cpp AND NOT TARGET yaml-cpp)
  add_library(yaml-cpp ALIAS yaml-cpp::yaml-cpp)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
find_dependency(CLI11 HINTS ${CMAKE_CURRENT_LIST_DIR}/../CLI11)
find_dependency(dlpack HINTS ${CMAKE_CURRENT_LIST_DIR}/../dlpack)
find_dependency(concurrentqueue HINTS ${CMAKE_CURRENT_LIST_DIR}/../concurrentqueue)
if(NOT Eigen3_ROOT)
  set(Eigen3_ROOT "${PACKAGE_PREFIX_DIR}/include/3rdparty/Eigen/include")
endif()
find_dependency(Eigen3 3.4)
find_dependency(fmt HINTS ${CMAKE_CURRENT_LIST_DIR}/../fmt)
find_dependency(magic_enum HINTS ${CMAKE_CURRENT_LIST_DIR}/../magic_enum)
find_dependency(nvtx3 HINTS ${CMAKE_CURRENT_LIST_DIR}/../nvtx3)
find_dependency(spdlog HINTS ${CMAKE_CURRENT_LIST_DIR}/../spdlog)
find_dependency(tl-expected HINTS ${CMAKE_CURRENT_LIST_DIR}/../tl-expected)
find_dependency(ucx HINTS ${CMAKE_CURRENT_LIST_DIR}/../ucx)

find_dependency(rapids_logger HINTS ${CMAKE_CURRENT_LIST_DIR}/../rapids_logger)
find_dependency(rmm HINTS ${CMAKE_CURRENT_LIST_DIR}/../rmm)
find_dependency(MATX HINTS ${CMAKE_CURRENT_LIST_DIR}/../matx)

# Optional dependency on UCXX for holoscan:pose_tree module
if(NOT TARGET ucxx::ucxx)
  if(holoscan_FIND_QUIETLY)
    set(holoscan_ucxx_FIND_QUIET "QUIET")
  endif()
  find_package(ucxx HINTS ${CMAKE_CURRENT_LIST_DIR}/../ucxx ${ucxx_FIND_QUIET})
endif()

# Optional dependency on Fast DDS and TinyXML2 for holoscan:ipc (holo_ipc).
# Always QUIET: absence is normal when IPC examples are off or CMAKE_PREFIX_PATH omits Fast DDS;
# holoscan-config.cmake already prepends this directory to CMAKE_MODULE_PATH for FindTinyXML2.cmake.
if(NOT TARGET fastdds)
  find_package(fastdds CONFIG QUIET)
endif()
if(NOT TARGET TinyXML2::TinyXML2)
  find_package(TinyXML2 CONFIG QUIET)
  if(NOT TinyXML2_FOUND)
    find_package(TinyXML2 MODULE QUIET)
  endif()
endif()

set(_GXF_components @HOLOSCAN_GXF_COMPONENTS@)

foreach(gxf_component IN LISTS _GXF_components)
  if(NOT TARGET GXF::${gxf_component})
    add_library(GXF::${gxf_component} SHARED IMPORTED)
    set_target_properties(GXF::${gxf_component} PROPERTIES
      IMPORTED_LOCATION "${PACKAGE_PREFIX_DIR}/lib/libgxf_${gxf_component}.so"
      IMPORTED_NO_SONAME ON
      INTERFACE_INCLUDE_DIRECTORIES "${PACKAGE_PREFIX_DIR}/include;${PACKAGE_PREFIX_DIR}/include/gxf"
    )
  endif()
endforeach()

@HOLOSCAN_GXF_INSTALL_HOOK@

set(GXF_LIB_DIR "${PACKAGE_PREFIX_DIR}/lib")
set(GXF_EXTENSIONS_DIR "${PACKAGE_PREFIX_DIR}/lib/gxf_extensions")

# Add the directory containing cmake files
# to the search paths so that find_dependency can find it.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

]=])
string(CONFIGURE "${holoscan_install_hook_code_string}" holoscan_install_hook_code_string @ONLY)

set(holoscan_build_hook_code_string [=[
]=])


# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_export.html#rapids-export
# (rapids_export() always installs to lib and doesn't use GNUInstallDirs)
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
# # Configure CPack packaging
# ##############################################################################
include(HoloscanCPack)
