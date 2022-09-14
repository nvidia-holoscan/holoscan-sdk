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

# create component library
add_library(holoviz_viewer_lib SHARED
  holoviz_viewer.cpp
  holoviz_viewer.hpp
)
target_link_libraries(holoviz_viewer_lib
  PUBLIC
  CUDA::cudart
  GXF::std
  GXF::serialization
  GXF::cuda
  yaml-cpp
  clara_holoviz
  clara_holoviz_imgui
)

# Create extension
add_library(holoviz_viewer SHARED
  holoviz_viewer_ext.cpp
)
target_link_libraries(holoviz_viewer
  PUBLIC
  holoviz_viewer_lib
)

# Install GXF extension as a component 'holoscan-embedded-gxf_extensions'
install_gxf_extension(holoviz_viewer)
