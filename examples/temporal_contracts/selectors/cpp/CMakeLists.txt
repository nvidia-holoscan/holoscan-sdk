# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_contract_selectors LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(contract_selectors contract_selectors.cpp)
target_link_libraries(contract_selectors PRIVATE holoscan::core)
holoscan_configure_example_install(contract_selectors)

if(BUILD_TESTING)
    # REQ: SYSTEM-013
    add_test(NAME EXAMPLE_CPP_CONTRACT_SELECTORS_TEST COMMAND contract_selectors)
    set_tests_properties(EXAMPLE_CPP_CONTRACT_SELECTORS_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "\\[selectors\\] assertions=pass"
            LABELS "examples;temporal-contract"
            TIMEOUT 30)
endif()

install(TARGETS contract_selectors
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/temporal_contracts/selectors/cpp"
    COMPONENT holoscan-examples
)
