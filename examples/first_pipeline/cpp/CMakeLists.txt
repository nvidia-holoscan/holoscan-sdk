# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_first_pipeline LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(first_pipeline first_pipeline.cpp)
target_link_libraries(first_pipeline PRIVATE holoscan::core)
holoscan_configure_example_install(first_pipeline)

if(BUILD_TESTING)
    # REQ: SYSTEM-013
    add_test(NAME EXAMPLE_CPP_FIRST_PIPELINE_TEST COMMAND first_pipeline)
    set_tests_properties(EXAMPLE_CPP_FIRST_PIPELINE_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "first_pipeline: received 10 doubled ticks \\(last=20\\) termination=stopped"
            LABELS "examples;authoring;graph"
            TIMEOUT 30)
endif()

install(TARGETS first_pipeline
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/first_pipeline/cpp"
    COMPONENT holoscan-examples
)
