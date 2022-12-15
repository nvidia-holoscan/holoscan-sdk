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

add_library(multiai_postprocessor_lib SHARED
  multiai_postprocessor.cpp
  multiai_postprocessor.hpp
)

target_link_libraries(multiai_postprocessor_lib
  PUBLIC
    CUDA::cudart
    GXF::cuda
    GXF::std
    GXF::multimedia
    yaml-cpp
    holoinfer
)

target_include_directories(multiai_postprocessor_lib
                  PUBLIC
                    ${CMAKE_SOURCE_DIR}/modules/holoinfer/src/include)

# Create extension
add_library(multiai_postprocessor SHARED
  multiai_postprocessor_extension.cpp
)
target_link_libraries(multiai_postprocessor
  PUBLIC multiai_postprocessor_lib
)

install_gxf_extension(multiai_postprocessor)
