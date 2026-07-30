#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ONNXRuntime REQUIRED)
add_library(holoinfer_ort SHARED core.cpp)
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
        dlpack::dlpack
        holoscan::logger
        holoscan::holoinfer_utils
        holoscan_security_flags
)
target_include_directories(holoinfer_ort
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
    )
