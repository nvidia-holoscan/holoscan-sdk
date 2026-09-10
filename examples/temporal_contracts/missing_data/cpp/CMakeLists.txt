# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_contract_missing_data LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(contract_missing_data contract_missing_data.cpp)
target_link_libraries(contract_missing_data PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(contract_missing_data)

if(BUILD_TESTING)
    # REQ: SYSTEM-013
    add_test(NAME EXAMPLE_CPP_CONTRACT_MISSING_DATA_TEST COMMAND contract_missing_data)
    set_tests_properties(EXAMPLE_CPP_CONTRACT_MISSING_DATA_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "missing_data: policy comparison complete"
            LABELS "examples;temporal-contract"
            TIMEOUT 30
    )
endif()

install(TARGETS contract_missing_data
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/temporal_contracts/missing_data/cpp"
    COMPONENT holoscan-examples
)
