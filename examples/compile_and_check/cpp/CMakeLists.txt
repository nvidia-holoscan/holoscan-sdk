# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_compile_and_check LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(compile_and_check compile_and_check.cpp)
target_link_libraries(compile_and_check PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(compile_and_check)

if(BUILD_TESTING)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../common/ctest_workspace.cmake")
    holoscan_configure_example_test_workspace(compile_and_check)
    # REQ: SYSTEM-014
    add_test(NAME EXAMPLE_CPP_COMPILE_AND_CHECK_TEST COMMAND compile_and_check)
    set_tests_properties(EXAMPLE_CPP_COMPILE_AND_CHECK_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "compile_and_check example complete: admission, diagnostics, and digest verified"
            LABELS "examples;compile"
            TIMEOUT 30)

    # Verify plan JSON remains stable across independent processes.
    add_test(
        NAME EXAMPLE_CPP_COMPILE_AND_CHECK_PLAN_JSON_DETERMINISM_TEST
        COMMAND ${CMAKE_COMMAND}
            "-DCOMPILE_AND_CHECK_EXECUTABLE=$<TARGET_FILE:compile_and_check>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/check_plan_json_determinism.cmake"
    )
    set_tests_properties(EXAMPLE_CPP_COMPILE_AND_CHECK_PLAN_JSON_DETERMINISM_TEST
        PROPERTIES
            LABELS "examples;compile"
            TIMEOUT 90)
endif()

install(TARGETS compile_and_check
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/compile_and_check/cpp"
    COMPONENT holoscan-core-examples
)

add_executable(diagnostic_reporting diagnostic_reporting.cpp)
target_link_libraries(diagnostic_reporting PRIVATE holoscan::core)
if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_DIAGNOSTIC_REPORTING_TEST COMMAND diagnostic_reporting)
    set_tests_properties(EXAMPLE_CPP_DIAGNOSTIC_REPORTING_TEST PROPERTIES
        PASS_REGULAR_EXPRESSION "diagnostic_reporting example complete: caller-owned startup failure report"
        LABELS "examples;compile" TIMEOUT 30)
endif()
holoscan_configure_example_install(diagnostic_reporting)
install(TARGETS diagnostic_reporting
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/compile_and_check/cpp"
    COMPONENT holoscan-core-examples)
