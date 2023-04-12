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
add_library(gxf_tensor_rt_lib SHARED
  tensor_rt_inference.cpp
  tensor_rt_inference.hpp
)
target_link_libraries(gxf_tensor_rt_lib
  PUBLIC
    CUDA::cudart
    GXF::cuda
    GXF::std
    TensorRT::nvonnxparser
    yaml-cpp
)

# Create extension
add_library(gxf_tensor_rt SHARED
  tensor_rt_extension.cpp
)
target_link_libraries(gxf_tensor_rt
  PUBLIC gxf_tensor_rt_lib
)

# Install GXF extension as a component 'holoscan-gxf_extensions'
install_gxf_extension(gxf_tensor_rt)
