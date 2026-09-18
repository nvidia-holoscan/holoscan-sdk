# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_reusable_graphs LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(reusable_graphs reusable_graphs.cpp)
target_compile_features(reusable_graphs PRIVATE cxx_std_20)
target_link_libraries(reusable_graphs PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(reusable_graphs)

if(BUILD_TESTING)
    # REQ: SYSTEM-014
    add_test(NAME EXAMPLE_CPP_REUSABLE_GRAPHS_TEST COMMAND reusable_graphs)
    set_tests_properties(EXAMPLE_CPP_REUSABLE_GRAPHS_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "reusable_graphs PASS: direct-stage and wrapped-stage/inner replayed twice region=stage termination=stopped"
            LABELS "examples;graph"
            TIMEOUT 30)
endif()

install(TARGETS reusable_graphs
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/reusable_graphs/cpp"
    COMPONENT holoscan-core-examples
)
