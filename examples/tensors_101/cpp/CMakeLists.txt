# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_tensors_101 LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(tensors_101 tensors_101.cpp)
target_link_libraries(tensors_101 PRIVATE holoscan::core)
holoscan_configure_example_install(tensors_101)

if(BUILD_TESTING)
    # REQ: SYSTEM-013
    add_test(NAME EXAMPLE_CPP_TENSORS_101_TEST COMMAND tensors_101)
    set_tests_properties(EXAMPLE_CPP_TENSORS_101_TEST
        PROPERTIES PASS_REGULAR_EXPRESSION "tensors_101: round trip complete"
                   LABELS "examples;tensor"
                   TIMEOUT 30)
    # REQ: SYSTEM-013
    add_test(NAME EXAMPLE_CPP_TENSORS_101_EXHAUST_TEST COMMAND tensors_101 --exhaust)
    set_tests_properties(EXAMPLE_CPP_TENSORS_101_EXHAUST_TEST
        PROPERTIES PASS_REGULAR_EXPRESSION "tensors_101: exhaust demo complete"
                   LABELS "examples;tensor"
                   TIMEOUT 30)
endif()

install(TARGETS tensors_101
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/tensors_101/cpp"
    COMPONENT holoscan-examples
)
