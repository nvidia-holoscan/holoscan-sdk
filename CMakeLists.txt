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

set(HOLOSCAN_PACKAGE_NAME holoscan-embedded)

# ##############################################################################
# # Options
# ##############################################################################
option(BUILD_SHARED_LIBS "Build Shared Libraries" ON)
option(HOLOSCAN_BUILD_MODULES "Build Holoscan Embedded SDK Modules" ON)
option(HOLOSCAN_BUILD_GXF_EXTENSIONS "Build GXF Extensions" ON)
option(HOLOSCAN_BUILD_APPS "Build Holoscan Embedded SDK Apps" ON)
option(HOLOSCAN_DOWNLOAD_DATASETS "Download SDK Apps Datasets" ON)
option(HOLOSCAN_BUILD_TESTS "Build Holoscan Embedded SDK Tests" ON)
option(HOLOSCAN_BUILD_DOCS "Build Holoscan Embedded SDK Documents" OFF)
option(HOLOSCAN_USE_CCACHE "Use ccache for building Holoscan Embedded SDK" ON)
option(HOLOSCAN_BUILD_HI_SPEED_ENDO_APP "Build Holoscan Hi-Speed Endoscopy App" OFF)

set(HOLOSCAN_CACHE_DIR ".cache" CACHE PATH "Directory to use for caching data for CPM and CCache. If relative, it is relative to the source directory (CMAKE_SOURCE_DIR).")
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
    DESCRIPTION "Holoscan Embedded SDK"
    LANGUAGES C CXX
)

# Delayed CUDA language enablement (to set CMAKE_CUDA_ARCHITECTURES and CMAKE_CUDA_HOST_COMPILER)
enable_language(CUDA)

# ##############################################################################
# # Default language settings
# ##############################################################################
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED YES)
SET(CMAKE_CXX_EXTENSIONS NO)

# ##############################################################################
# # RAPIDS CMAKE configurations
# # (https://docs.rapids.ai/api/rapids-cmake/stable/api.html)
# ##############################################################################

# Set default build type in case not specified
rapids_cmake_build_type(release)

# Generate a C++ header file (under ${CMAKE_CURRENT_BINARY_DIR}) that hold the version (X.Y.Z) information of the calling project
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
add_library(${HOLOSCAN_PACKAGE_NAME}
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
    include/holoscan/core/gxf/gxf_operator.hpp
    include/holoscan/core/gxf/gxf_resource.hpp
    include/holoscan/core/gxf/gxf_utils.hpp
    include/holoscan/core/io_context.hpp
    include/holoscan/core/io_spec.hpp
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
    include/holoscan/operators/custom_lstm_inference/lstm_tensor_rt_inference.hpp
    include/holoscan/operators/format_converter/format_converter.hpp
    include/holoscan/operators/stream_playback/video_stream_recorder.hpp
    include/holoscan/operators/stream_playback/video_stream_replayer.hpp
    include/holoscan/operators/tensor_rt/tensor_rt_inference.hpp
    include/holoscan/operators/video_composer/video_composer.hpp
    include/holoscan/operators/visualizer_tool_tracking/visualizer_tool_tracking.hpp
    include/holoscan/std_ops.hpp
    include/holoscan/utils/yaml_parser.hpp
    src/core/application.cpp
    src/core/arg.cpp
    src/core/condition.cpp
    src/core/conditions/gxf/boolean.cpp
    src/core/conditions/gxf/count.cpp
    src/core/conditions/gxf/downstream_affordable.cpp
    src/core/conditions/gxf/message_available.cpp
    src/core/config.cpp
    src/core/executors/gxf/gxf_executor.cpp
    src/core/fragment.cpp
    src/core/graphs/flow_graph.cpp
    src/core/gxf/gxf_condition.cpp
    src/core/gxf/gxf_operator.cpp
    src/core/gxf/gxf_resource.cpp
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
    src/operators/custom_lstm_inference/lstm_tensor_rt_inference.cpp
    src/operators/format_converter/format_converter.cpp
    src/operators/stream_playback/video_stream_recorder.cpp
    src/operators/stream_playback/video_stream_replayer.cpp
    src/operators/tensor_rt/tensor_rt_inference.cpp
    src/operators/visualizer_tool_tracking/visualizer_tool_tracking.cpp
)

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

# Note: Looks like the following line causes error on CMake 3.18.4 (it is working on 3.18.2). Keeping it for now.
set(HOLOSCAN_REQUIRED_FEATURES cxx_std_17)
target_compile_features(${HOLOSCAN_PACKAGE_NAME} PRIVATE ${HOLOSCAN_REQUIRED_FEATURES})

# Use generator expression to avoid `nvcc fatal   : Value '-std=c++17' is not defined for option 'Werror'`
target_compile_options(${HOLOSCAN_PACKAGE_NAME}
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Werror -Wall -Wextra>
)

# Link libraries
target_link_libraries(${HOLOSCAN_PACKAGE_NAME}
    PUBLIC
    ${CMAKE_DL_LIBS}
    Threads::Threads
    fmt::fmt-header-only
    GXF::core
    yaml-cpp

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
    COMPONENT holoscan-embedded-core
)

# Copy headers
install(DIRECTORY include/holoscan/
    DESTINATION include/holoscan
    COMPONENT holoscan-embedded-core
)

# Copy bundled tl-expected headers
install(DIRECTORY ${tl-expected_SOURCE_DIR}/include/tl/
    DESTINATION include/3rdparty/tl
    COMPONENT holoscan-embedded-core
)

# Copy version file
install(FILES ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/include/holoscan/version_config.hpp
    DESTINATION include/holoscan
    COMPONENT holoscan-embedded-core
)

# Copy FindXXX files generated by CPM
install(FILES
    ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/CPM_modules/Findajantv2.cmake
    ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/CPM_modules/Findfmt.cmake
    ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/CPM_modules/FindGLAD.cmake
    ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/CPM_modules/FindGLFW.cmake
    ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/CPM_modules/Findnanovg.cmake
    ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/CPM_modules/Findspdlog.cmake
    ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/CPM_modules/Findtl-expected.cmake
    ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/CPM_modules/Findyaml-cpp.cmake
    DESTINATION cmake
    COMPONENT holoscan-embedded-core
)

# Copy FindGXF module
install(FILES ${CMAKE_CURRENT_LIST_DIR}/cmake/modules/FindGXF.cmake
    DESTINATION cmake
    COMPONENT holoscan-embedded-core
)

# Define docs and hooks
set(holoscan_doc_string [=[
libholoscan-embedded: Holoscan Embedded SDK C++ API
]=])

set(holoscan_install_hook_code_string [=[

if(NOT GXF_SDK_PATH)
    message(FATAL_ERROR "Please set 'GXF_SDK_PATH' variable in CMake")
else()
    list(APPEND CMAKE_PREFIX_PATH "${GXF_SDK_PATH}")
    message("GXF_SDK_PATH is set to ${GXF_SDK_PATH}")
endif()

find_package(GXF 2.4 MODULE REQUIRED
  COMPONENTS
    core
)

]=])

set(holoscan_build_hook_code_string [=[
]=])

# Override CMAKE_INSTALL_DEFAULT_COMPONENT_NAME to hholoscan-embedded-core
# until rapids-cmake's rapids_export include the concept of optional COMPONENTS
# (https://github.com/rapidsai/rapids-cmake/pull/154)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME holoscan-embedded-core)

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
# # Build Apps
# ##############################################################################
if(HOLOSCAN_BUILD_APPS)
  add_subdirectory(apps)
endif()

# ##############################################################################
# # Add tests
# ##############################################################################
if(HOLOSCAN_BUILD_TESTS AND CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Enable testing for the current directory and below
    include(CTest) # it calls 'enable_testing()' internally
    superbuild_depend(gtest_rapids)

    add_subdirectory(tests)
endif()

# ##############################################################################
# # Add documentation
# ##############################################################################
if(HOLOSCAN_BUILD_DOCS)
    add_subdirectory(docs)
endif()
