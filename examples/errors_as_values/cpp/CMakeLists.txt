# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_errors_as_values LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(errors_as_values errors_as_values.cpp)
target_link_libraries(errors_as_values PRIVATE holoscan::core)
holoscan_configure_example_install(errors_as_values)

if(BUILD_TESTING)
    # REQ: SYSTEM-014
    add_test(NAME EXAMPLE_CPP_ERRORS_AS_VALUES_TEST COMMAND errors_as_values)
    set_tests_properties(EXAMPLE_CPP_ERRORS_AS_VALUES_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "errors_as_values: transient_retries=3 delivered=3 \\(last=30\\) run_diagnostics=0 termination=stopped"
            LABELS "examples;error-handling"
            TIMEOUT 30)
endif()

install(TARGETS errors_as_values
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/errors_as_values/cpp"
    COMPONENT holoscan-examples
)
