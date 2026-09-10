#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
project(holoscan_inference_toolkit
    VERSION ${HOLOSCAN_BUILD_VERSION}
    DESCRIPTION "Holoscan Inference toolkit"
    LANGUAGES CXX CUDA
    )

add_subdirectory(src)
add_subdirectory(operators)

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

install(EXPORT ${HOLOSCAN_PACKAGE_NAME}-inference-exports
    FILE ${HOLOSCAN_PACKAGE_NAME}-inference-targets.cmake
    NAMESPACE ${HOLOSCAN_PACKAGE_NAME}::
    DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}/cmake/${HOLOSCAN_PACKAGE_NAME}
    COMPONENT holoscan-inference
)
