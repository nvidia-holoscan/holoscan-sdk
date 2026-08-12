# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_compile_and_check LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(compile_and_check compile_and_check.cpp)
target_link_libraries(compile_and_check PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(compile_and_check)

if(BUILD_TESTING)
    # REQ: SYSTEM-014
    add_test(NAME EXAMPLE_CPP_COMPILE_AND_CHECK_TEST COMMAND compile_and_check)
    set_tests_properties(EXAMPLE_CPP_COMPILE_AND_CHECK_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "compile_and_check example complete: admission, diagnostics, and digest verified"
            LABELS "examples;compile"
            TIMEOUT 30)
endif()

install(TARGETS compile_and_check
    RUNTIME DESTINATION examples/compile_and_check/cpp
    COMPONENT holoscan-examples
)
