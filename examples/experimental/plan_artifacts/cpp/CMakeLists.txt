# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_plan_artifacts LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(plan_artifacts plan_artifacts.cpp)
target_link_libraries(plan_artifacts PRIVATE holoscan::core)
holoscan_configure_example_install(plan_artifacts)

if(BUILD_TESTING)
    # REQ: SYSTEM-015
    add_test(NAME EXAMPLE_CPP_PLAN_ARTIFACTS_TEST COMMAND plan_artifacts)
    set_tests_properties(EXAMPLE_CPP_PLAN_ARTIFACTS_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "plan_artifacts example complete: accessors, digest, and idempotent artifacts verified"
            LABELS "examples;graph-artifacts;experimental"
            TIMEOUT 30)

    add_executable(plan_artifacts_workspace_test plan_artifacts_workspace_test.cpp)
    add_test(
        NAME EXAMPLE_CPP_PLAN_ARTIFACTS_WORKSPACE_COLLISION_TEST
        COMMAND plan_artifacts_workspace_test)
    set_tests_properties(EXAMPLE_CPP_PLAN_ARTIFACTS_WORKSPACE_COLLISION_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "plan_artifacts workspace collision isolation verified"
            LABELS "examples;graph-artifacts;experimental"
            TIMEOUT 30)

    add_test(
        NAME EXAMPLE_CPP_PLAN_ARTIFACTS_UNAVAILABLE_TEMP_ROOT_TEST
        COMMAND
            ${CMAKE_COMMAND}
            -DPLAN_ARTIFACTS_EXAMPLE=$<TARGET_FILE:plan_artifacts>
            -DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/plan_artifacts_temp_root_test.cmake)
    set_tests_properties(EXAMPLE_CPP_PLAN_ARTIFACTS_UNAVAILABLE_TEMP_ROOT_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "plan_artifacts unavailable temporary root handled without an exception"
            LABELS "examples;graph-artifacts;experimental"
            TIMEOUT 30)
endif()

install(TARGETS plan_artifacts
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/experimental/plan_artifacts/cpp"
    COMPONENT holoscan-core-examples
)
