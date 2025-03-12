# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
add_library(gxf_ucx_holoscan_lib SHARED
  ucx_holoscan_component_serializer.cpp
)
target_link_libraries(gxf_ucx_holoscan_lib
  PUBLIC
    CUDA::cudart
    GXF::std
    GXF::multimedia
    GXF::serialization
    ucx::ucp
    yaml-cpp
    holoscan::core  # needed for ucx_holoscan_component_serializer.cpp
)

# Create extension
add_library(gxf_ucx_holoscan SHARED
  ucx_holoscan_ext.cpp
)
target_compile_definitions(gxf_ucx_holoscan
  PRIVATE HOLOSCAN_BUILD_VERSION=${HOLOSCAN_BUILD_VERSION}
)
target_link_libraries(gxf_ucx_holoscan
  PUBLIC  gxf_ucx_holoscan_lib
  PRIVATE holoscan_security_flags
)

if(HOLOSCAN_REGISTER_GXF_EXTENSIONS)
  include(WrapOperatorAsGXFExtension)
  generate_gxf_registry_manifest(
    EXTENSION_TARGET gxf_ucx_holoscan
    EXTENSION_NAME UcxHoloscanExtension
    FORWARD_ARGS
    --uuid "e549f7ce-9ecf-4d53-8156-418727c176df"
    --version "${holoscan_VERSION}"
    --extension-dependencies
      $<TARGET_FILE:GXF::std>
      $<TARGET_FILE:GXF::multimedia>
      $<TARGET_FILE:GXF::serialization>
    --headers
      "${CMAKE_CURRENT_SOURCE_DIR}/ucx_holoscan_component_serializer.hpp"
  )
  register_gxf_extension(
    EXTENSION_NAME UcxHoloscanExtension
    MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/UcxHoloscanExtension_manifest.yaml"
  )
endif()

# Install GXF extension as a component 'holoscan-gxf_extensions'
install_gxf_extension(gxf_ucx_holoscan)
