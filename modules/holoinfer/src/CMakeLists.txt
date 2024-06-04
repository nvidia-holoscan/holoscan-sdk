#
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if(HOLOSCAN_BUILD_LIBTORCH)
    find_package(Torch REQUIRED)
    find_package(TorchVision REQUIRED)
    add_library(holoinfer_torch SHARED infer/torch/core.cpp)
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
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${dlpack_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${magic_enum_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
        )
endif()

if(HOLOSCAN_BUILD_ORT)
    find_package(ONNXRuntime REQUIRED)
    add_library(holoinfer_ort SHARED infer/onnx/core.cpp)
    add_library(holoscan::infer::onnx_runtime ALIAS holoinfer_ort)
    set_target_properties(holoinfer_ort PROPERTIES
        OUTPUT_NAME holoscan_infer_onnx_runtime
        EXPORT_NAME infer::onnx_runtime
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION}
        INSTALL_RPATH_USE_LINK_PATH true # find onnxruntime in install tree
    )
    target_link_libraries(holoinfer_ort
        PRIVATE
            ONNXRuntime::ONNXRuntime
            CUDA::cudart
            holoscan::logger
            GXF::core
            yaml-cpp
            holoscan_security_flags
    )
    target_include_directories(holoinfer_ort
        PRIVATE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
        )
endif()

set(holoinfer_src
    infer/trt/core.cpp
    infer/trt/utils.cpp
    params/infer_param.cpp
    process/data_processor.cpp
    process/transforms/generate_boxes.cpp
    manager/infer_manager.cpp
    manager/process_manager.cpp
    utils/infer_utils.cpp
    utils/infer_buffer.cpp
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
        yaml-cpp
    PRIVATE
        ${CMAKE_DL_LIBS} # to dlopen holoinfer_torch
        CUDA::cudart
        holoscan::logger
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
