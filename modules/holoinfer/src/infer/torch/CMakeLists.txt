#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

# It is necessary to set the TORCH_CUDA_ARCH_LIST explicitly for newer versions of Torch (circa v2.3.0)
# Note: `find_package(Torch)` is overwriting CUDA_ARCHITECTURES, make sure to pay attention to this
# when adding CUDA kernels to this the Torch backend library.
set(TORCH_CUDA_ARCH_LIST "5.0 8.0 8.6 8.9 9.0")
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
add_library(holoinfer_torch SHARED core.cpp)
add_library(holoscan::infer::torch ALIAS holoinfer_torch)
set_target_properties(holoinfer_torch PROPERTIES
    OUTPUT_NAME holoscan_infer_torch
    EXPORT_NAME infer::torch
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    INSTALL_RPATH_USE_LINK_PATH true # find libtorch and torchvision in install tree
)
target_link_libraries(holoinfer_torch
    PRIVATE
        torch
        -Wl,--no-as-needed TorchVision::TorchVision # torch plugin
        holoscan::logger
        GXF::core
        yaml-cpp
        holoscan_security_flags
)
target_include_directories(holoinfer_torch
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<BUILD_INTERFACE:${dlpack_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
    )
