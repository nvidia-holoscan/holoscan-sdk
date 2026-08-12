# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_embedded_device_tensor LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(embedded_device_tensor embedded_device_tensor.cpp)
target_compile_features(embedded_device_tensor PRIVATE cxx_std_20)
target_link_libraries(embedded_device_tensor
    PRIVATE
        CUDA::cudart
        holoscan::core
        Threads::Threads
)
holoscan_configure_example_install(embedded_device_tensor)

if(BUILD_TESTING)
    # REQ: COMPILE-048
    # REQ: TENSOR-CORE-013
    # REQ: SYSTEM-018
    # REQ: SYSTEM-020
    add_test(NAME EXAMPLE_CPP_EMBEDDED_DEVICE_TENSOR_TEST COMMAND embedded_device_tensor)
    set_tests_properties(EXAMPLE_CPP_EMBEDDED_DEVICE_TENSOR_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "embedded_device_tensor PASS: flat-equivalent reusable plan replayed twice"
            SKIP_RETURN_CODE 77
            LABELS "examples;graph;tensor;cuda;deployment"
            TIMEOUT 30
    )

    # REQ: SYSTEM-018
    # REQ: SYSTEM-020
    add_test(
        NAME EXAMPLE_CPP_EMBEDDED_DEVICE_TENSOR_ARTIFACT_TEST
        COMMAND embedded_device_tensor
            --emit-artifacts "${CMAKE_CURRENT_BINARY_DIR}/test-artifacts")
    set_tests_properties(EXAMPLE_CPP_EMBEDDED_DEVICE_TENSOR_ARTIFACT_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "artifact-only publication complete"
            LABELS "examples;graph;artifacts;deployment"
            RUN_SERIAL TRUE
            TIMEOUT 15
    )
endif()

install(TARGETS embedded_device_tensor
    RUNTIME DESTINATION examples/embedded_device_tensor/cpp
    COMPONENT holoscan-examples
)
