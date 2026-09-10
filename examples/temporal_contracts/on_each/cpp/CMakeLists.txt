# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_contract_on_each LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(contract_on_each contract_on_each.cpp)
target_link_libraries(contract_on_each PRIVATE holoscan::core)
holoscan_configure_example_install(contract_on_each)

if(BUILD_TESTING)
    # REQ: SYSTEM-013
    add_test(NAME EXAMPLE_CPP_CONTRACT_ON_EACH_TEST COMMAND contract_on_each)
    set_tests_properties(EXAMPLE_CPP_CONTRACT_ON_EACH_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "on_each contract demo PASS: anchor-rate sampling with reuse, stale side suppressed"
            LABELS "examples;temporal-contract"
            TIMEOUT 30)
endif()

install(TARGETS contract_on_each
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/temporal_contracts/on_each/cpp"
    COMPONENT holoscan-examples
)
