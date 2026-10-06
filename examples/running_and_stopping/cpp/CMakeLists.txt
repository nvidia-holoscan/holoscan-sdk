# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_running_and_stopping LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(running_and_stopping running_and_stopping.cpp)
target_link_libraries(running_and_stopping PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(running_and_stopping)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_RUNNING_AND_STOPPING_TEST COMMAND running_and_stopping)
    set_tests_properties(EXAMPLE_CPP_RUNNING_AND_STOPPING_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "running_and_stopping: all sub-runs behaved as documented"
            LABELS "examples;run-session"
            TIMEOUT 30
    )
endif()

install(TARGETS running_and_stopping
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/running_and_stopping/cpp"
    COMPONENT holoscan-core-examples
)
