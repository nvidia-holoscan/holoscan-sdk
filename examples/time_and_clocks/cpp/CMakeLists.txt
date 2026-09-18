# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_time_and_clocks LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

find_package(Threads REQUIRED)

add_executable(time_and_clocks time_and_clocks.cpp)
target_link_libraries(time_and_clocks PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(time_and_clocks)

if(BUILD_TESTING)
    # REQ: SYSTEM-014
    add_test(NAME EXAMPLE_CPP_TIME_AND_CLOCKS_TEST COMMAND time_and_clocks)
    set_tests_properties(EXAMPLE_CPP_TIME_AND_CLOCKS_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "time_and_clocks\\[realtime\\]: hw accepted=11 dropped=1 act accepted=12 forward_preserved=yes termination=stopped"
            LABELS "examples;time"
            TIMEOUT 30)
    # REQ: SYSTEM-014
    add_test(NAME EXAMPLE_CPP_TIME_AND_CLOCKS_MANUAL_TEST COMMAND time_and_clocks --manual)
    set_tests_properties(EXAMPLE_CPP_TIME_AND_CLOCKS_MANUAL_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "time_and_clocks\\[manual\\]: hw accepted=11 dropped=1 act accepted=12 forward_preserved=yes termination=stopped"
            LABELS "examples;time"
            TIMEOUT 30)
endif()

install(TARGETS time_and_clocks
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/time_and_clocks/cpp"
    COMPONENT holoscan-core-examples
)
