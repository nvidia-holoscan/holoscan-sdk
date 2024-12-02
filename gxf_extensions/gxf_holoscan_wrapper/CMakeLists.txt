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

# Create library
add_library(gxf_holoscan_wrapper_lib SHARED
  operator_wrapper.cpp
  operator_wrapper.hpp
  operator_wrapper_fragment.cpp
  operator_wrapper_fragment.hpp
)
target_link_libraries(gxf_holoscan_wrapper_lib
  PUBLIC
    holoscan::core
)

# Create extension
add_library(gxf_holoscan_wrapper SHARED
  gxf_holoscan_wrapper_ext.cpp
)
target_compile_definitions(gxf_holoscan_wrapper
  PRIVATE HOLOSCAN_BUILD_VERSION=${HOLOSCAN_BUILD_VERSION}
)
target_link_libraries(gxf_holoscan_wrapper
  PUBLIC gxf_holoscan_wrapper_lib
  PRIVATE holoscan_security_flags
)

# Add include directories
target_include_directories(gxf_holoscan_wrapper_lib
    PUBLIC
    # Include headers from the parent directory (./gxf_extensions)
    # so that the headers can be included as <gxf_holoscan_wrapper/operator_wrapper.hpp>
    # TODO(unknown): need to find better way to do this
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Generate the GXF registry manifest
include(WrapOperatorAsGXFExtension)
generate_gxf_registry_manifest(
  EXTENSION_TARGET gxf_holoscan_wrapper
  EXTENSION_NAME HoloscanWrapperExtension
  BINARY_FILES
    $<TARGET_FILE:gxf_holoscan_wrapper>
    $<TARGET_FILE:gxf_holoscan_wrapper_lib>
    $<TARGET_FILE:GXF::logger>
    $<TARGET_FILE:holoscan::core>
    $<TARGET_FILE:holoscan::infer>
    $<TARGET_FILE:holoscan::infer::onnx_runtime>
    $<TARGET_FILE:holoscan::infer::torch>
    $<TARGET_FILE:holoscan::logger>
    $<TARGET_FILE:holoscan::ops::aja>
    $<TARGET_FILE:holoscan::ops::async_ping_rx>
    $<TARGET_FILE:holoscan::ops::async_ping_tx>
    $<TARGET_FILE:holoscan::ops::bayer_demosaic>
    $<TARGET_FILE:holoscan::ops::format_converter>
    $<TARGET_FILE:holoscan::ops::gxf_codelet>
    $<TARGET_FILE:holoscan::ops::holoviz>
    $<TARGET_FILE:holoscan::ops::inference>
    $<TARGET_FILE:holoscan::ops::inference_processor>
    $<TARGET_FILE:holoscan::ops::ping_rx>
    $<TARGET_FILE:holoscan::ops::ping_tensor_rx>
    $<TARGET_FILE:holoscan::ops::ping_tensor_tx>
    $<TARGET_FILE:holoscan::ops::ping_tx>
    $<TARGET_FILE:holoscan::ops::segmentation_postprocessor>
    $<TARGET_FILE:holoscan::ops::v4l2>
    $<TARGET_FILE:holoscan::ops::video_stream_recorder>
    $<TARGET_FILE:holoscan::ops::video_stream_replayer>
    $<TARGET_FILE:holoscan::spdlog_logger>
    $<TARGET_FILE:holoscan::viz>
    $<TARGET_FILE:ucx::ucp>
    $<TARGET_FILE:ucx::ucs>
    $<TARGET_FILE:ucx::uct>
    $<TARGET_FILE:yaml-cpp::yaml-cpp>
  FORWARD_ARGS
    --uuid "12d01b4e-e06f-49ef-93c4-961834347385"
    --version "${holoscan_VERSION}"
    --extension-dependencies
      $<TARGET_FILE:GXF::std>
      $<TARGET_FILE:GXF::cuda>
      $<TARGET_FILE:GXF::multimedia>
      $<TARGET_FILE:GXF::serialization>
      $<TARGET_FILE:GXF::ucx>
      $<TARGET_FILE:gxf_ucx_holoscan>
    --headers
      "${CMAKE_CURRENT_SOURCE_DIR}/operator_wrapper.hpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/operator_wrapper_fragment.hpp"
)
if(HOLOSCAN_REGISTER_GXF_EXTENSIONS)
  register_gxf_extension(
    EXTENSION_NAME HoloscanWrapperExtension
    MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/HoloscanWrapperExtension_manifest.yaml"
    DEPENDS register_UcxHoloscanExtension
  )
endif()

# Install the header files
install(FILES
  operator_wrapper.hpp
  operator_wrapper_fragment.hpp
  # Install headers to the directory 'gxf_holoscan_wrapper' in the include directory
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gxf_holoscan_wrapper
  COMPONENT holoscan-gxf_extensions
)

# Install GXF extension as a component 'holoscan-gxf_extensions'
install_gxf_extension(gxf_holoscan_wrapper)
