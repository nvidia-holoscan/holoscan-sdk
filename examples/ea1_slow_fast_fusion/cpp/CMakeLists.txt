# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_ea1_slow_fast_fusion LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(ea1_slow_fast_fusion slow_fast_fusion.cpp)
target_compile_features(ea1_slow_fast_fusion PRIVATE cxx_std_20)
target_link_libraries(ea1_slow_fast_fusion PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(ea1_slow_fast_fusion)

holoscan_add_graph_artifacts(
    TARGET ea1_slow_fast_fusion
    DEPLOYMENT_ID developer
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/holoscan-artifacts"
)

if(BUILD_TESTING)
    # REQ: SYSTEM-004
    add_test(NAME EXAMPLE_EA1_SLOW_FAST_FUSION_TEST COMMAND ea1_slow_fast_fusion)
    set_tests_properties(EXAMPLE_EA1_SLOW_FAST_FUSION_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "slow/fast fusion complete"
            TIMEOUT 15
    )
endif()

install(TARGETS ea1_slow_fast_fusion
    RUNTIME DESTINATION examples/ea1_slow_fast_fusion/cpp
    COMPONENT holoscan-examples
)
