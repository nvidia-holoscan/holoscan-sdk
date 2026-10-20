# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_determinism_replay LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(determinism_replay determinism_replay.cpp)
target_link_libraries(determinism_replay PRIVATE holoscan::core)
holoscan_configure_example_install(determinism_replay)

if(BUILD_TESTING)
    # REQ: SYSTEM-015
    add_test(NAME EXAMPLE_CPP_DETERMINISM_REPLAY_TEST COMMAND determinism_replay)
    set_tests_properties(EXAMPLE_CPP_DETERMINISM_REPLAY_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "determinism_replay: manual_replay_match=yes realtime_tagtime_differs=yes wall_clock_replayed=no ticks=6 termination=stopped"
            LABELS "examples;deterministic;experimental"
            TIMEOUT 30)
endif()

install(TARGETS determinism_replay
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/experimental/determinism_replay/cpp"
    COMPONENT holoscan-core-examples
)
