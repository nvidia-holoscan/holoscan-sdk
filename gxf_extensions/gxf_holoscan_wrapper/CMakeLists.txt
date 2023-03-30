# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
target_link_libraries(gxf_holoscan_wrapper
  PUBLIC gxf_holoscan_wrapper_lib
)

# Add include directories
target_include_directories(gxf_holoscan_wrapper_lib
    PUBLIC
    # Include headers from the parent directory (./gxf_extensions)
    # so that the headers can be included as <gxf_holoscan_wrapper/operator_wrapper.hpp>
    # TODO: need to find better way to do this
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

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
