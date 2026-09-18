# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_first_operator LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(first_operator first_operator.cpp)
target_link_libraries(first_operator PRIVATE holoscan::core)
holoscan_configure_example_install(first_operator)

if(BUILD_TESTING)
    # REQ: SYSTEM-013
    add_test(NAME EXAMPLE_CPP_FIRST_OPERATOR_TEST COMMAND first_operator)
    set_tests_properties(EXAMPLE_CPP_FIRST_OPERATOR_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "first_operator complete: doubled=[1-9][0-9]* fused_fresh=[1-9][0-9]* fused_reused=[1-9][0-9]* fused_missing=[0-9]+"
            LABELS "examples;authoring"
            TIMEOUT 30)
endif()

install(TARGETS first_operator
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/first_operator/cpp"
    COMPONENT holoscan-core-examples
)
