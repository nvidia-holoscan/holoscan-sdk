# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_contract_on_any_input LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(contract_on_any_input contract_on_any_input.cpp)
target_link_libraries(contract_on_any_input PRIVATE holoscan::core)
holoscan_configure_example_install(contract_on_any_input)

if(BUILD_TESTING)
    # REQ: SYSTEM-013
    add_test(NAME EXAMPLE_CPP_CONTRACT_ON_ANY_INPUT_TEST COMMAND contract_on_any_input)
    set_tests_properties(EXAMPLE_CPP_CONTRACT_ON_ANY_INPUT_TEST
        PROPERTIES PASS_REGULAR_EXPRESSION
            "every activation was attributed to exactly one input port"
        LABELS "examples;temporal-contract"
        TIMEOUT 30)
endif()

install(TARGETS contract_on_any_input
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/temporal_contracts/on_any_input/cpp"
    COMPONENT holoscan-examples
)
