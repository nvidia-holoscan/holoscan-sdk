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

add_subdirectory(include) # header-only interface for public holoinfer headers (e.g., ActivationSpec)
add_subdirectory(infer) # inference backends

set(holoinfer_utils_src
    utils/infer_utils.cpp
    utils/infer_buffer.cpp
    utils/work_queue.cpp
)
add_library(holoinfer_utils SHARED ${holoinfer_utils_src})
add_library(holoscan::holoinfer_utils ALIAS holoinfer_utils)

target_include_directories(holoinfer_utils
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(holoinfer_utils
    PUBLIC
        CUDA::cuda_driver
        CUDA::nvrtc
        holoscan::infer::headers
    PRIVATE
        CUDA::cudart
        holoscan_security_flags
        yaml-cpp
)

set_target_properties(holoinfer_utils PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)

set(holoinfer_src
    infer/trt/core.cpp
    infer/trt/utils.cpp
    params/infer_param.cpp
    process/data_processor.cpp
    process/data_processor.cu
    process/transforms/generate_boxes.cpp
    manager/infer_manager.cpp
    manager/process_manager.cpp
)

add_library(${PROJECT_NAME} SHARED ${holoinfer_src})
add_library(holoscan::infer ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PUBLIC
        $<BUILD_INTERFACE:${tl-expected_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${dlpack_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${magic_enum_SOURCE_DIR}/include>
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        CUDA::cuda_driver
        CUDA::nvrtc
        holoscan::infer::headers
        holoscan::holoinfer_utils # [1]
    PRIVATE
        ${CMAKE_DL_LIBS} # to dlopen holoinfer_torch
        CUDA::cudart
        holoscan::core # [2]
        holoscan::logger
        TensorRT::nvinfer_plugin
        TensorRT::nvonnxparser
        holoscan_security_flags
        yaml-cpp
)
# [1] holoinfer_utils is technically a PRIVATE dependency only as we don't use any types from it
#     through the public header of holoinfer/holoscan::infer (holoinfer.hpp). However, the
#     holoinfer_utils sources used to belong to holoscan::infer and we want to keep them as a
#     PUBLIC dependency to avoid breaking changes, so that people linking against holoscan::infer
#     today can still use holoscan::holoinfer_utils symbols without changing their CMake. If you
#     happen to only use symbols from holoscan::holoinfer_utils and not use symbols from
#     holoscan::infer anymore, you can switch your dependency from the latter to the former.
# [2] This target - holoinfer/holoscan::infer - was meant to be holoscan agnostic so that one could
#     use it without pulling the holoscan SDK core (schedulers, etc.), just like holoviz, which is
#     why the sources are under `modules`. This is the reason why we have a target in the top `src`
#     CMakeLists.txt named holoscan::infer_utils (to not be confused with holoscan::holoinfer_utils)
#     to handle the inference-specific utility functions that depend on both holoscan::core and
#     holoinfer which holoscan inference operators can them leverage. However, at the time of writing
#     holoinfer is currently depending on holoscan::DataExporter which is part of the
#     holoscan::core library (core/analytics), so the architecture does not match the intended
#     design today. Future changes could include moving the DataExporter to a separate
#     library and target, or embracing the symbiotic link between holoinfer and holoscan core, and
#     get rid of the holoscan::infer_utils target which could be merged back in holoinfer. The point
#     of this comment was to highlight the current architecture which can appear confusing otherwise.

if(HOLOSCAN_BUILD_LIBTORCH)
    # no target_link_libraries to holoinfer_torch: we dlopen it as a plugin
    target_compile_definitions(${PROJECT_NAME} PUBLIC "HOLOINFER_TORCH_ENABLED")
endif()
if(HOLOSCAN_BUILD_ORT)
    # no target_link_libraries to holoinfer_ort: we dlopen it as a plugin
    target_compile_definitions(${PROJECT_NAME} PUBLIC "HOLOINFER_ORT_ENABLED")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME holoscan_infer
    EXPORT_NAME infer
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)

install(TARGETS holoinfer_utils
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
