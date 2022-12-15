#
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
#


file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION HOLOINFER_VERSION)

project(holoinfer VERSION ${HOLOINFER_VERSION} LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(ONNXRuntime REQUIRED)

add_library(holoinfer SHARED
    infer/onnx/core.cpp
    infer/trt/core.cpp
    infer/trt/utils.cpp
    params/infer_param.cpp
    process/data_processor.cpp
    manager/infer_manager.cpp
    manager/process_manager.cpp
    utils/infer_utils.cpp
    utils/infer_buffer.cpp)

target_include_directories(holoinfer
                        PRIVATE 
                            $<BUILD_INTERFACE:${ONNXRuntime_INCLUDE_DIR}>
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                        PUBLIC
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include 
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                        )
                           
target_link_libraries(holoinfer PUBLIC 
                      ONNXRuntime::ONNXRuntime
                      CUDA::cudart
                      GXF::cuda
                      GXF::std
                      GXF::multimedia
                      yaml-cpp
                      CUDA::nppidei
                      CUDA::nppig
                      TensorRT::nvonnxparser
                      )

# install the ONNX runtime library
get_target_property(ONNXRuntimeLibLocation ONNXRuntime::ONNXRuntime LOCATION)
get_filename_component(ONNXRuntimeLocation ${ONNXRuntimeLibLocation} REALPATH)
install(FILES "${ONNXRuntimeLocation}"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT "holoscan-dependencies"
)

install(TARGETS holoinfer
    EXPORT ${PROJECT_NAME}Config
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# TODO: Commenting the install since it's included in the SDK
#install(EXPORT ${PROJECT_NAME}Config
#    DESTINATION cmake
#)