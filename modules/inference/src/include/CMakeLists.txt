#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

find_package(CUDAToolkit REQUIRED)

# Header-only interface for public holoinfer headers without large dependencies
add_library(holoinfer_headers INTERFACE)
add_library(holoscan::infer::headers ALIAS holoinfer_headers)
set_target_properties(holoinfer_headers PROPERTIES EXPORT_NAME infer::headers)

add_library(holoinfer_headers_compat INTERFACE)
add_library(holoscan::holoinfer_headers ALIAS holoinfer_headers_compat)
set_target_properties(holoinfer_headers_compat PROPERTIES EXPORT_NAME holoinfer_headers)
target_link_libraries(holoinfer_headers_compat INTERFACE holoscan::infer::headers)
target_include_directories(holoinfer_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        # Generated holoinfer/holoinfer_export.h lives at
        # `<build>/modules/inference/src/include/holoinfer/holoinfer_export.h`, i.e. here.
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(holoinfer_headers
    INTERFACE
        CUDA::cudart
        holoscan::logger
)

# Install. Sources live under `.../include/holoinfer/*.hpp` so consumers write
# `#include <holoinfer/foo.hpp>` (natural include root, matching Core / HoloMQ).
install(TARGETS holoinfer_headers holoinfer_headers_compat
    EXPORT ${HOLOSCAN_PACKAGE_NAME}-inference-exports
    COMPONENT ${HOLOSCAN_INFERENCE_INSTALL_COMPONENT}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/holoinfer
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ${HOLOSCAN_INFERENCE_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.hpp"
)
