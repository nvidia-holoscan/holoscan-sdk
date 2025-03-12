#
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
#

project(holoinfer VERSION ${HOLOSCAN_BUILD_VERSION} LANGUAGES CXX CUDA)

add_subdirectory(infer)

find_package(CUDAToolkit REQUIRED)

set(holoinfer_src
    infer/trt/core.cpp
    infer/trt/utils.cpp
    params/infer_param.cpp
    process/data_processor.cpp
    process/data_processor.cu
    process/transforms/generate_boxes.cpp
    manager/infer_manager.cpp
    manager/process_manager.cpp
    utils/infer_utils.cpp
    utils/infer_buffer.cpp
    utils/work_queue.cpp
)

add_library(${PROJECT_NAME} SHARED ${holoinfer_src})
add_library(holoscan::infer ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PUBLIC
        $<BUILD_INTERFACE:${tl-expected_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${dlpack_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${magic_enum_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        GXF::core
        GXF::cuda
        GXF::multimedia
        GXF::std
        CUDA::cuda_driver
        CUDA::nvrtc
        yaml-cpp
    PRIVATE
        ${CMAKE_DL_LIBS} # to dlopen holoinfer_torch
        CUDA::cudart
        holoscan::logger
        TensorRT::nvinfer_plugin
        TensorRT::nvonnxparser
        holoscan_security_flags
)

if(HOLOSCAN_BUILD_LIBTORCH)
    # no target_link_libraries to holoinfer_torch: we dlopen it as a plugin
    target_compile_definitions(${PROJECT_NAME} PUBLIC "use_torch=true")
endif()
if(HOLOSCAN_BUILD_ORT)
    # no target_link_libraries to holoinfer_ort: we dlopen it as a plugin
    target_compile_definitions(${PROJECT_NAME} PUBLIC "use_onnxruntime=true")
endif()

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
