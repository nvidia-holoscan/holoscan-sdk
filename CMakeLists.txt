# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# CUDA_STANDARD 17 is supported from CMAKE 3.18
# : https://cmake.org/cmake/help/v3.18/prop_tgt/CUDA_STANDARD.html
cmake_minimum_required(VERSION 3.20.1) # rapids-cmake-src requires >= 3.20.1

set(HOLOSCAN_PACKAGE_NAME holoscan)

# ##############################################################################
# # Options
# ##############################################################################
option(BUILD_SHARED_LIBS "Build Shared Libraries" ON)
option(HOLOSCAN_BUILD_MODULES "Build Holoscan SDK Modules" ON)
option(HOLOSCAN_BUILD_GXF_EXTENSIONS "Build GXF Extensions" ON)
option(HOLOSCAN_BUILD_APPS "Build Holoscan SDK Apps" ON)
option(HOLOSCAN_BUILD_EXAMPLES "Build Holoscan SDK Examples" ON)
option(HOLOSCAN_BUILD_PYTHON "Build Holoscan SDK Python Bindings" ON)
option(HOLOSCAN_DOWNLOAD_DATASETS "Download SDK Apps Datasets" ON)
option(HOLOSCAN_BUILD_TESTS "Build Holoscan SDK Tests" ON)
option(HOLOSCAN_BUILD_DOCS "Build Holoscan SDK Documents" OFF)
option(HOLOSCAN_USE_CCACHE "Use ccache for building Holoscan SDK" ON)
option(HOLOSCAN_BUILD_HI_SPEED_ENDO_APP "Build Holoscan Hi-Speed Endoscopy App" OFF)

set(HOLOSCAN_CACHE_DIR ".cache" CACHE PATH "Directory to use for caching data for CPM and CCache.\
                                            If relative, it is relative to the source directory (CMAKE_SOURCE_DIR).")
mark_as_advanced(HOLOSCAN_CACHE_DIR)

# HOLOSCAN_USE_CCACHE_SKIPPED will be set to TRUE by 'include(SetupCache)'
# later in the build process, if CCache is not available.
set(HOLOSCAN_USE_CCACHE_SKIPPED FALSE)

# ##############################################################################
# # Prerequisite statements
# ##############################################################################

# Set the top-level directory of the project
set(HOLOSCAN_TOP "${CMAKE_CURRENT_LIST_DIR}")

# Append local cmake module path for CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH "${HOLOSCAN_TOP}/cmake/modules")

# Append local cmake dependency path CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${HOLOSCAN_TOP}/cmake/deps")

# Set HOLOSCAN_BUILD_VERSION from 'VERSION' file
unset(HOLOSCAN_BUILD_VERSION CACHE)
file(STRINGS ${HOLOSCAN_TOP}/VERSION HOLOSCAN_BUILD_VERSION)

# ##############################################################################
# # Include build utilities
# ##############################################################################
include(SuperBuildUtils) # set local dependent modules and fetch RAPIDS-cmake

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# Configuring CUDA Architecture/Compiler with delayed CUDA configuration
# Need to be called before project()
# Since we want to set CMAKE_CUDA_HOST_COMPILER to use CMAKE_CXX_COMPILER for non-Clang compilers,
# we first call project() without 'CUDA' and then set CMAKE_CUDA_HOST_COMPILER before calling 'enable_language(CUDA)'
include(SetupCUDA)

# ##############################################################################
# # Project definition
# ##############################################################################
project(${HOLOSCAN_PACKAGE_NAME}
    VERSION ${HOLOSCAN_BUILD_VERSION}
    DESCRIPTION "Holoscan SDK"
    LANGUAGES C CXX
)

# Setting the default paths for the build tree
# Libraries (shared and static) are put by default in the lib directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set the default RPATH for built libraries to $ORIGIN and the lib directory
set(CMAKE_BUILD_RPATH \$ORIGIN:${CMAKE_BINARY_DIR}/lib)

# Delayed CUDA language enablement (to set CMAKE_CUDA_ARCHITECTURES and CMAKE_CUDA_HOST_COMPILER)
enable_language(CUDA)

# ##############################################################################
# # Default language settings
# ##############################################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

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
include(SetupCache)

# ##############################################################################
# # Define dependencies
# ##############################################################################
include(DefineSearchDirFor)
include(GenerateGXEApp)

include(cmake/setup_dependencies.cmake)

# ##############################################################################
# # Add library: ${HOLOSCAN_PACKAGE_NAME}
# ##############################################################################
set(LIST_SOURCES
    include/holoscan/core/application.hpp
    include/holoscan/core/arg.hpp
    include/holoscan/core/argument_setter.hpp
    include/holoscan/core/common.hpp
    include/holoscan/core/component_spec.hpp
    include/holoscan/core/component_spec-inl.hpp
    include/holoscan/core/component.hpp
    include/holoscan/core/condition.hpp
    include/holoscan/core/conditions/gxf/boolean.hpp
    include/holoscan/core/conditions/gxf/count.hpp
    include/holoscan/core/conditions/gxf/downstream_affordable.hpp
    include/holoscan/core/conditions/gxf/message_available.hpp
    include/holoscan/core/config.hpp
    include/holoscan/core/domain/tensor.hpp
    include/holoscan/core/execution_context.hpp
    include/holoscan/core/executor.hpp
    include/holoscan/core/executors/gxf/gxf_executor.hpp
    include/holoscan/core/executors/gxf/gxf_parameter_adaptor.hpp
    include/holoscan/core/expected.hpp
    include/holoscan/core/forward_def.hpp
    include/holoscan/core/fragment.hpp
    include/holoscan/core/graph.hpp
    include/holoscan/core/graphs/flow_graph.hpp
    include/holoscan/core/gxf/entity.hpp
    include/holoscan/core/gxf/gxf_component.hpp
    include/holoscan/core/gxf/gxf_execution_context.hpp
    include/holoscan/core/gxf/gxf_extension_registrar.hpp
    include/holoscan/core/gxf/gxf_io_context.hpp
    include/holoscan/core/gxf/gxf_operator.hpp
    include/holoscan/core/gxf/gxf_resource.hpp
    include/holoscan/core/gxf/gxf_tensor.hpp
    include/holoscan/core/gxf/gxf_utils.hpp
    include/holoscan/core/gxf/gxf_wrapper.hpp
    include/holoscan/core/io_context.hpp
    include/holoscan/core/io_spec.hpp
    include/holoscan/core/message.hpp
    include/holoscan/core/operator_spec.hpp
    include/holoscan/core/operator.hpp
    include/holoscan/core/parameter.hpp
    include/holoscan/core/resource.hpp
    include/holoscan/core/resources/gxf/allocator.hpp
    include/holoscan/core/resources/gxf/block_memory_pool.hpp
    include/holoscan/core/resources/gxf/cuda_stream_pool.hpp
    include/holoscan/core/resources/gxf/double_buffer_receiver.hpp
    include/holoscan/core/resources/gxf/double_buffer_transmitter.hpp
    include/holoscan/core/resources/gxf/receiver.hpp
    include/holoscan/core/resources/gxf/std_component_serializer.hpp
    include/holoscan/core/resources/gxf/transmitter.hpp
    include/holoscan/core/resources/gxf/unbounded_allocator.hpp
    include/holoscan/core/resources/gxf/video_stream_serializer.hpp
    include/holoscan/core/type_traits.hpp
    include/holoscan/holoscan.hpp
    include/holoscan/operators/aja_source/aja_source.hpp
    include/holoscan/operators/aja_source/ntv2channel.hpp
    include/holoscan/operators/bayer_demosaic/bayer_demosaic.hpp
    include/holoscan/operators/custom_lstm_inference/lstm_tensor_rt_inference.hpp
    include/holoscan/operators/format_converter/format_converter.hpp
    include/holoscan/operators/holoviz/holoviz.hpp
    include/holoscan/operators/multiai_inference/multiai_inference.hpp
    include/holoscan/operators/multiai_postprocessor/multiai_postprocessor.hpp
    include/holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp
    include/holoscan/operators/stream_playback/video_stream_recorder.hpp
    include/holoscan/operators/stream_playback/video_stream_replayer.hpp
    include/holoscan/operators/tensor_rt/tensor_rt_inference.hpp
    include/holoscan/operators/tool_tracking_postprocessor/tool_tracking_postprocessor.hpp
    include/holoscan/operators/visualizer_icardio/visualizer_icardio.hpp
    include/holoscan/operators/visualizer_tool_tracking/visualizer_tool_tracking.hpp
    include/holoscan/std_ops.hpp
    include/holoscan/utils/timer.hpp
    include/holoscan/utils/yaml_parser.hpp
    src/core/application.cpp
    src/core/arg.cpp
    src/core/argument_setter.cpp
    src/core/condition.cpp
    src/core/conditions/gxf/boolean.cpp
    src/core/conditions/gxf/count.cpp
    src/core/conditions/gxf/downstream_affordable.cpp
    src/core/conditions/gxf/message_available.cpp
    src/core/config.cpp
    src/core/domain/tensor.cpp
    src/core/executors/gxf/gxf_executor.cpp
    src/core/executors/gxf/gxf_parameter_adaptor.cpp
    src/core/fragment.cpp
    src/core/graphs/flow_graph.cpp
    src/core/gxf/entity.cpp
    src/core/gxf/gxf_condition.cpp
    src/core/gxf/gxf_execution_context.cpp
    src/core/gxf/gxf_io_context.cpp
    src/core/gxf/gxf_operator.cpp
    src/core/gxf/gxf_resource.cpp
    src/core/gxf/gxf_tensor.cpp
    src/core/gxf/gxf_wrapper.cpp
    src/core/io_spec.cpp
    src/core/logger.cpp
    src/core/operator.cpp
    src/core/resource.cpp
    src/core/resources/gxf/block_memory_pool.cpp
    src/core/resources/gxf/cuda_stream_pool.cpp
    src/core/resources/gxf/double_buffer_receiver.cpp
    src/core/resources/gxf/double_buffer_transmitter.cpp
    src/core/resources/gxf/std_component_serializer.cpp
    src/core/resources/gxf/video_stream_serializer.cpp
    src/operators/aja_source/aja_source.cpp
    src/operators/bayer_demosaic/bayer_demosaic.cpp
    src/operators/custom_lstm_inference/lstm_tensor_rt_inference.cpp
    src/operators/format_converter/format_converter.cpp
    src/operators/holoviz/holoviz.cpp
    src/operators/multiai_inference/multiai_inference.cpp
    src/operators/multiai_postprocessor/multiai_postprocessor.cpp
    src/operators/segmentation_postprocessor/segmentation_postprocessor.cpp
    src/operators/stream_playback/video_stream_recorder.cpp
    src/operators/stream_playback/video_stream_replayer.cpp
    src/operators/tensor_rt/tensor_rt_inference.cpp
    src/operators/tool_tracking_postprocessor/tool_tracking_postprocessor.cpp
    src/operators/visualizer_icardio/visualizer_icardio.cpp
    src/operators/visualizer_tool_tracking/visualizer_tool_tracking.cpp
)

if(HOLOSCAN_BUILD_HI_SPEED_ENDO_APP)
    set(LIST_SOURCES
        ${LIST_SOURCES}
        include/holoscan/operators/emergent_source/emergent_source.hpp
        src/operators/emergent_source/emergent_source.cpp
  )
endif()

add_library(${HOLOSCAN_PACKAGE_NAME} ${LIST_SOURCES})

# Compile options
set_target_properties(${HOLOSCAN_PACKAGE_NAME}
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO

    # https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
    INSTALL_RPATH \$ORIGIN
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)

# Force targets using holoscan SDK to use the C++17 standard
target_compile_features(${HOLOSCAN_PACKAGE_NAME} INTERFACE cxx_std_17)

# Note: Looks like the following line causes error on CMake 3.18.4 (it is working on 3.18.2). Keeping it for now.
set(HOLOSCAN_REQUIRED_FEATURES cxx_std_17)
target_compile_features(${HOLOSCAN_PACKAGE_NAME} PRIVATE ${HOLOSCAN_REQUIRED_FEATURES})

# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(${HOLOSCAN_PACKAGE_NAME}
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>
)

# Build EVT related files if HOLOSCAN_BUILD_HI_SPEED_ENDO_APP is ON.
if(HOLOSCAN_BUILD_HI_SPEED_ENDO_APP)
    target_compile_definitions(${HOLOSCAN_PACKAGE_NAME} PUBLIC HOLOSCAN_BUILD_EMERGENT=1)
endif()

# Skip trace logging in release build if CMAKE_BUILD_TYPE is Release.
# (See include/holoscan/core/logger.hpp)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(${HOLOSCAN_PACKAGE_NAME} PRIVATE HOLOSCAN_LOG_ACTIVE_LEVEL=1)
endif()

# Link libraries
target_link_libraries(${HOLOSCAN_PACKAGE_NAME}
    PUBLIC
    ${CMAKE_DL_LIBS}
    Threads::Threads
    fmt::fmt-header-only
    yaml-cpp
    GXF::core
    GXF::std
    CUDA::cudart

    PRIVATE
    spdlog::spdlog_header_only
)

target_include_directories(${HOLOSCAN_PACKAGE_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${HOLOSCAN_TOP}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>

    # Bundled tl-expected
    $<BUILD_INTERFACE:${tl-expected_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/3rdparty>

    # Bundled dlpack
    $<BUILD_INTERFACE:${dlpack_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/3rdparty>

    PRIVATE
    ${HOLOSCAN_TOP}/src
)

# ##############################################################################
# # Package project
# ##############################################################################
set(HOLOSCAN_INSTALL_TARGETS
    ${HOLOSCAN_PACKAGE_NAME}
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
DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
COMPONENT "holoscan-gxf_libs"
)

# Define docs and hooks
set(holoscan_doc_string [=[
libholoscan: Holoscan SDK C++ API
]=])


# Defines the install export hook
# We use add_library since we are installing the libraries as part of the SDK
set(holoscan_install_hook_code_string

[=[

add_library(fmt::fmt-header-only INTERFACE IMPORTED)

set(_GXFlibs core std multimedia cuda network npp serialization behavior_tree)

foreach(gxflib IN LISTS _GXFlibs)
  add_library(GXF::${gxflib} SHARED IMPORTED)
  set_target_properties(GXF::${gxflib} PROPERTIES
    IMPORTED_LOCATION "${PACKAGE_PREFIX_DIR}/lib/libgxf_${gxflib}.so"
    IMPORTED_NO_SONAME ON
    INTERFACE_INCLUDE_DIRECTORIES "${PACKAGE_PREFIX_DIR}/include;${PACKAGE_PREFIX_DIR}/include/gxf"
  )
endforeach()

add_executable(GXF::gxe IMPORTED)
set_target_properties(GXF::gxe PROPERTIES
  IMPORTED_LOCATION "${PACKAGE_PREFIX_DIR}/bin/gxe"
)

set(HOLOSCAN_PACKAGE_NAME holoscan::holoscan)
set(GXF_LIB_DIR "${PACKAGE_PREFIX_DIR}/lib")
set(GXF_EXTENSIONS_DIR "${PACKAGE_PREFIX_DIR}/lib/gxf_extensions")

include("${CMAKE_CURRENT_LIST_DIR}/GenerateGXEAppInstall.cmake")

]=])

set(holoscan_build_hook_code_string [=[
]=])

# Override CMAKE_INSTALL_DEFAULT_COMPONENT_NAME to holoscan-core
# until rapids-cmake's rapids_export include the concept of optional COMPONENTS
# (https://github.com/rapidsai/rapids-cmake/pull/154)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME holoscan-core)

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_export.html#rapids-export
# (rapids_export() always installs to lib and doesn’t use GNUInstallDirs)
rapids_export(
    INSTALL ${HOLOSCAN_PACKAGE_NAME}
    EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    GLOBAL_TARGETS ${HOLOSCAN_INSTALL_TARGETS}
    NAMESPACE ${HOLOSCAN_PACKAGE_NAME}::
    DOCUMENTATION holoscan_doc_string
    FINAL_CODE_BLOCK holoscan_install_hook_code_string
)
rapids_export(
    BUILD ${HOLOSCAN_PACKAGE_NAME}
    EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    GLOBAL_TARGETS ${HOLOSCAN_INSTALL_TARGETS}
    LANGUAGES C CXX CUDA
    NAMESPACE ${HOLOSCAN_PACKAGE_NAME}::
    DOCUMENTATION holoscan_doc_string
    FINAL_CODE_BLOCK holoscan_build_hook_code_string
)

# ##############################################################################
# # Build Coverage
# ##############################################################################
if(HOLOSCAN_BUILD_COVERAGE)
    include(cmake/modules/CodeCoverage.cmake)
    append_coverage_compiler_flags()
    setup_target_for_coverage_lcov(
        NAME coverage
	BASE_DIRECTORY "${CMAKE_SOURCE_DIR}"
	CTEST_EXECUTABLE ctest --test-dir ${CMAKE_BINARY_DIR} || true
	CTEST_EXTRACT "${CMAKE_SOURCE_DIR}/gxf_extensions/*"
        "${CMAKE_SOURCE_DIR}/include/holoscan/*"
        "${CMAKE_SOURCE_DIR}/modules/holoviz/src/*"
        "${CMAKE_SOURCE_DIR}/modules/holoinfer/src/*"
        "${CMAKE_SOURCE_DIR}/src/*"
        "${CMAKE_SOURCE_DIR}/python/pybind11/*"
	PYTEST_EXECUTABLE pytest --pyargs "${CMAKE_BINARY_DIR}/python/lib/tests" || true
	PYTEST_EXTRACT "*holoscan*"
	)
endif()

# ##############################################################################
# # Build sub modules (holoviz)
# ##############################################################################
if(HOLOSCAN_BUILD_MODULES)
    add_subdirectory(modules)
endif()

# ##############################################################################
# # Build GXF Extensions and apps
# ##############################################################################
if(HOLOSCAN_BUILD_GXF_EXTENSIONS)
    add_subdirectory(gxf_extensions)
endif()

# ##############################################################################
# # Download datasets
# ##############################################################################
if(HOLOSCAN_DOWNLOAD_DATASETS)
  add_subdirectory(data)
endif()

# ##############################################################################
# # Build apps
# ##############################################################################
if(HOLOSCAN_BUILD_APPS)
  add_subdirectory(apps)
endif()

# ##############################################################################
# # Build examples
# ##############################################################################
if(HOLOSCAN_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ##############################################################################
# # Add tests
# ##############################################################################
if(HOLOSCAN_BUILD_TESTS)
    # Enable testing for the current directory and below
    include(CTest) # it calls 'enable_testing()' internally
    superbuild_depend(gtest_rapids)

    add_subdirectory(tests)

    # add Holoviz tests
    add_test(NAME HOLOVIZ_FUNCTIONAL_TEST COMMAND holoscan::viz::functionaltests)
    add_test(NAME HOLOVIZ_UNIT_TEST COMMAND holoscan::viz::unittests)
endif()

if(HOLOSCAN_BUILD_PYTHON)
    find_package(Python3  REQUIRED COMPONENTS Interpreter Development)
    superbuild_depend(pybind11)
    add_subdirectory(python)
endif()

# ##############################################################################
# # Add documentation
# ##############################################################################
if(HOLOSCAN_BUILD_DOCS)
    add_subdirectory(docs)
endif()

include(HoloscanCPack)
