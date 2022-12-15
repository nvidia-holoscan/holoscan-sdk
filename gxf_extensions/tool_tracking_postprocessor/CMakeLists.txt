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

# Mark CUDA files
# Note: needed because we can't use .cu or .cuh with the
# bazel build system. We could switch to using those instead
# if we only support CMake
set_source_files_properties(
  tool_tracking_postprocessor.cu.cpp
  tool_tracking_postprocessor.cu.hpp
  PROPERTIES
    LANGUAGE CUDA
)

# Create library
add_library(tool_tracking_postprocessor_lib SHARED
  tool_tracking_postprocessor.cpp
  tool_tracking_postprocessor.cu.cpp
  tool_tracking_postprocessor.cu.hpp
  tool_tracking_postprocessor.hpp
)
target_link_libraries(tool_tracking_postprocessor_lib
  PUBLIC
    CUDA::cuda_driver
    GXF::multimedia
    GXF::std
    yaml-cpp
)

# Create extension
add_library(tool_tracking_postprocessor SHARED
  tool_tracking_postprocessor_ext.cpp
)
target_link_libraries(tool_tracking_postprocessor
  PUBLIC tool_tracking_postprocessor_lib
)
# Install GXF extension as a component 'holoscan-gxf_extensions'
install_gxf_extension(tool_tracking_postprocessor)
