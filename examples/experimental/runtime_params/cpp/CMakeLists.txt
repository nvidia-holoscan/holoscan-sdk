# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_runtime_params LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(runtime_params runtime_params.cpp)
target_link_libraries(runtime_params PRIVATE holoscan::core)
holoscan_configure_example_install(runtime_params)

if(BUILD_TESTING)
    # REQ: SYSTEM-015
    add_test(NAME EXAMPLE_CPP_RUNTIME_PARAMS_TEST COMMAND runtime_params)
    set_tests_properties(EXAMPLE_CPP_RUNTIME_PARAMS_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "runtime_params: gain 1->2 generation=1 state=observed mismatch=rejected events=seen fence=closed termination=stopped"
            LABELS "examples;parameter;runtime-update;experimental"
            TIMEOUT 30)
endif()

install(TARGETS runtime_params
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/experimental/runtime_params/cpp"
    COMPONENT holoscan-examples
)
