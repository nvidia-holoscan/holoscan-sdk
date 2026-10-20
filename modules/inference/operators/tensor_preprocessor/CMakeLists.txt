# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(op_tensor_preprocessor
    tensor_preprocessor.cpp
)
add_library(holoscan::op_tensor_preprocessor ALIAS op_tensor_preprocessor)

target_include_directories(op_tensor_preprocessor
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(op_tensor_preprocessor
    PUBLIC
        holoscan::core
        holoscan::sensor_io
    PRIVATE
        CUDA::cudart
        CUDA::nppc
        CUDA::nppig
        CUDA::nppidei
        CUDA::nppial
        # Color-conversion library for the YUV -> RGB path (nppiYCbCr422ToRGB_8u_C2C3R_Ctx).
        CUDA::nppicc
        $<BUILD_INTERFACE:yaml-cpp::yaml-cpp>
        $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:
            $<INSTALL_INTERFACE:holoscan::_inference_yaml_private>>
)

set_target_properties(op_tensor_preprocessor PROPERTIES
    OUTPUT_NAME holoscan_op_tensor_preprocessor
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)
# Keep static dependencies such as yaml-cpp out of the exported symbols, as other SDK libraries do.
holoscan_configure_linux_shared_link(op_tensor_preprocessor)

install(TARGETS op_tensor_preprocessor
    EXPORT ${HOLOSCAN_PACKAGE_NAME}-inference-exports
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT holoscan-inference
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/holoscan/operators/tensor_preprocessor
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/operators
    COMPONENT holoscan-inference
    FILES_MATCHING PATTERN "*.hpp"
)
