#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(CUDAToolkit REQUIRED)

# Header-only interface for public holoinfer headers without large dependencies
add_library(holoinfer_headers INTERFACE)
add_library(holoscan::infer::headers ALIAS holoinfer_headers)
target_include_directories(holoinfer_headers
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
)
target_link_libraries(holoinfer_headers
    INTERFACE
        CUDA::cudart
        holoscan::logger
)

# Install
install(TARGETS holoinfer_headers
    EXPORT ${HOLOSCAN_PACKAGE_NAME}-exports
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
