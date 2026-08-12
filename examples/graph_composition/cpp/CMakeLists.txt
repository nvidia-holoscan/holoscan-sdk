# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_graph_composition LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(graph_composition graph_composition.cpp)
target_link_libraries(graph_composition PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(graph_composition)

if(BUILD_TESTING)
    # REQ: SYSTEM-014
    add_test(NAME EXAMPLE_CPP_GRAPH_COMPOSITION_TEST COMMAND graph_composition)
    set_tests_properties(EXAMPLE_CPP_GRAPH_COMPOSITION_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "\\[graph_composition\\] result blocking_dropped=0 freshest_dropped=[1-9][0-9]*"
            LABELS "examples;graph"
            TIMEOUT 30
    )
endif()

install(TARGETS graph_composition
    RUNTIME DESTINATION examples/graph_composition/cpp
    COMPONENT holoscan-examples
)
