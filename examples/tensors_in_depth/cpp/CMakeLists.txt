# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_tensors_in_depth LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(tensors_in_depth tensors_in_depth.cpp)
target_compile_features(tensors_in_depth PRIVATE cxx_std_20)
target_link_libraries(tensors_in_depth
    PRIVATE
        CUDA::cudart
        holoscan::core
)
holoscan_configure_example_install(tensors_in_depth)

if(BUILD_TESTING)
    # REQ: SYSTEM-015
    # CUDA-gated: with a usable device the binary prints the full-run marker;
    # without one it probes cudaGetDeviceCount, prints the skip marker, and
    # exits 77 so CTest reports the row as skipped on CPU-only CI lanes.
    add_test(NAME EXAMPLE_CPP_TENSORS_IN_DEPTH_TEST COMMAND tensors_in_depth)
    set_tests_properties(EXAMPLE_CPP_TENSORS_IN_DEPTH_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "tensors_in_depth: device tour complete"
            SKIP_RETURN_CODE 77
            LABELS "examples;tensor"
            TIMEOUT 30
    )
endif()

install(TARGETS tensors_in_depth
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/tensors_in_depth/cpp"
    COMPONENT holoscan-examples
)
