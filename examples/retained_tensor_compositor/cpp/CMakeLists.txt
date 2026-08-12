# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_retained_tensor_compositor LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(retained_tensor_compositor retained_tensor_compositor.cpp)
target_compile_features(retained_tensor_compositor PRIVATE cxx_std_20)
target_link_libraries(retained_tensor_compositor
    PRIVATE
        CUDA::cudart
        holoscan::core
        Threads::Threads
)
holoscan_configure_example_install(retained_tensor_compositor)

if(BUILD_TESTING)
    # REQ: CHAN-078
    # REQ: CHAN-102
    # REQ: EXTERNAL-003
    # REQ: RUNTIME-017
    # REQ: RUNTIME-034
    # REQ: SYSTEM-018
    add_test(
        NAME EXAMPLE_RETAINED_TENSOR_COMPOSITOR_HOST
        COMMAND "${CMAKE_COMMAND}"
            "-DHOLOSCAN_RETAINED_TENSOR_COMPOSITOR_EXECUTABLE=$<TARGET_FILE:retained_tensor_compositor>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/run_host_and_route.cmake"
    )
    set_tests_properties(EXAMPLE_RETAINED_TENSOR_COMPOSITOR_HOST
        PROPERTIES
            LABELS "examples;tensor;temporal;runtime"
            PASS_REGULAR_EXPRESSION
                "retained Tensor compositor README host and authored-route modes passed"
            TIMEOUT 20
    )

    # REQ: COMPILE-048
    # REQ: SYSTEM-018
    add_test(
        NAME EXAMPLE_RETAINED_TENSOR_COMPOSITOR_CUDA
        COMMAND retained_tensor_compositor --cuda --device 0
    )
    set_tests_properties(EXAMPLE_RETAINED_TENSOR_COMPOSITOR_CUDA
        PROPERTIES
            LABELS "cuda;examples;tensor;temporal;runtime"
            PASS_REGULAR_EXPRESSION "retained Tensor compositor complete"
            SKIP_RETURN_CODE 77
            TIMEOUT 30
    )

    # REQ: SYSTEM-018
    add_test(
        NAME EXAMPLE_RETAINED_TENSOR_COMPOSITOR_INVALID_DEVICE
        COMMAND "${CMAKE_COMMAND}"
            "-DHOLOSCAN_RETAINED_TENSOR_COMPOSITOR_EXECUTABLE=$<TARGET_FILE:retained_tensor_compositor>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/run_invalid_device.cmake"
    )
    set_tests_properties(EXAMPLE_RETAINED_TENSOR_COMPOSITOR_INVALID_DEVICE
        PROPERTIES
            LABELS "cuda;diagnostics;examples;tensor"
            SKIP_RETURN_CODE 77
            TIMEOUT 30
    )
endif()

install(TARGETS retained_tensor_compositor
    RUNTIME DESTINATION examples/retained_tensor_compositor/cpp
    COMPONENT holoscan-examples
)
