# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_holoinfer_example LANGUAGES CXX CUDA)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(holoinfer_example holoinfer_example.cpp)
target_link_libraries(holoinfer_example
    PRIVATE
        holoscan::core
        holoscan::op_inference
        CUDA::cudart
)
holoscan_configure_example_install(holoinfer_example)

if(HOLOSCAN_BUILD_INFERENCE AND BUILD_TESTING)
    add_test(NAME HoloInferExample_Smoke
        COMMAND ${CMAKE_COMMAND}
            -DEXAMPLE_BIN=$<TARGET_FILE:holoinfer_example>
            -DCONFIG_DIR=${CMAKE_CURRENT_SOURCE_DIR}/configs
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_smoke_test.cmake
    )
    set_tests_properties(HoloInferExample_Smoke
        PROPERTIES
            LABELS "examples;inference"
            TIMEOUT 120)
endif()

install(TARGETS holoinfer_example
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/inference"
    COMPONENT holoscan-examples
)
install(DIRECTORY configs
    DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/inference"
    COMPONENT holoscan-examples
)
install(DIRECTORY models
    DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/inference"
    COMPONENT holoscan-examples
)
