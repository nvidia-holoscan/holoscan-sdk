# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_contract_deadlines LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(contract_deadlines contract_deadlines.cpp)
target_compile_features(contract_deadlines PRIVATE cxx_std_20)
target_link_libraries(contract_deadlines PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(contract_deadlines)

if(BUILD_TESTING)
    # REQ: SYSTEM-013
    add_test(NAME EXAMPLE_CPP_CONTRACT_DEADLINES_TEST COMMAND contract_deadlines)
    set_tests_properties(EXAMPLE_CPP_CONTRACT_DEADLINES_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "contract deadlines complete"
            LABELS "examples;temporal-contract"
            TIMEOUT 30
    )
    # The wrapper asserts both the deliberate nonzero exit and the exact success
    # token. PASS_REGULAR_EXPRESSION alone would ignore the process exit code.
    # REQ: SYSTEM-013
    add_test(NAME EXAMPLE_CPP_CONTRACT_DEADLINES_FAILRUN_TEST
        COMMAND "${CMAKE_COMMAND}"
            "-DPROGRAM=$<TARGET_FILE:contract_deadlines>"
            -P "${CMAKE_CURRENT_LIST_DIR}/verify_fail_run.cmake")
    set_tests_properties(EXAMPLE_CPP_CONTRACT_DEADLINES_FAILRUN_TEST
        PROPERTIES
            LABELS "examples;temporal-contract"
            TIMEOUT 30
    )
endif()

install(TARGETS contract_deadlines
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/temporal_contracts/deadlines/cpp"
    COMPONENT holoscan-core-examples
)
