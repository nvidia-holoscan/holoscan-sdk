#
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
#

project(holoinfer VERSION ${HOLOSCAN_BUILD_VERSION} LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(ONNXRuntime REQUIRED)

add_library(${PROJECT_NAME} SHARED
    infer/onnx/core.cpp
    infer/trt/core.cpp
    infer/trt/utils.cpp
    params/infer_param.cpp
    process/data_processor.cpp
    manager/infer_manager.cpp
    manager/process_manager.cpp
    utils/infer_utils.cpp
    utils/infer_buffer.cpp
    )
add_library(holoscan::infer ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
                        PRIVATE
                            $<BUILD_INTERFACE:${ONNXRuntime_INCLUDE_DIR}>
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                        PUBLIC
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
                        )

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        GXF::core
        GXF::cuda
        GXF::multimedia
        GXF::std
        yaml-cpp
    PRIVATE
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::nppidei
        CUDA::nppig
        holoscan::logger
        TensorRT::nvonnxparser
        ONNXRuntime::ONNXRuntime
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME holoscan_infer
    EXPORT_NAME infer
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
        )

# install the ONNX runtime library
get_target_property(ONNXRuntimeLibLocation ONNXRuntime::ONNXRuntime LOCATION)
get_filename_component(ONNXRuntimeLocation ${ONNXRuntimeLibLocation} REALPATH)

# export the name of the lib
get_filename_component(ONNXRuntimeLibName ${ONNXRuntimeLocation} NAME)
set(ONNXRUNTIME_LIBRARY_NAME ${ONNXRuntimeLibName} CACHE INTERNAL "ONNXRuntime library name")

install(FILES "${ONNXRuntimeLocation}"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT "holoscan-dependencies"
)
