# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_contract_max_start_delay LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(contract_max_start_delay contract_max_start_delay.cpp)
target_compile_features(contract_max_start_delay PRIVATE cxx_std_20)
target_link_libraries(contract_max_start_delay PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(contract_max_start_delay)

if(BUILD_TESTING)
    # Validate the example’s observable post-stop result.
    add_test(NAME EXAMPLE_CPP_CONTRACT_MAX_START_DELAY_TEST COMMAND contract_max_start_delay)
    set_tests_properties(EXAMPLE_CPP_CONTRACT_MAX_START_DELAY_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "contract max_start_delay complete"
            LABELS "examples;temporal-contract"
            TIMEOUT 30
    )
    # The wrapper asserts both the deliberate nonzero exit and the exact success
    # token. PASS_REGULAR_EXPRESSION alone would ignore the process exit code.
    # Validate the example’s observable post-stop result.
    add_test(NAME EXAMPLE_CPP_CONTRACT_MAX_START_DELAY_FAILRUN_TEST
        COMMAND "${CMAKE_COMMAND}"
            "-DPROGRAM=$<TARGET_FILE:contract_max_start_delay>"
            -P "${CMAKE_CURRENT_LIST_DIR}/verify_fail_run.cmake")
    set_tests_properties(EXAMPLE_CPP_CONTRACT_MAX_START_DELAY_FAILRUN_TEST
        PROPERTIES
            LABELS "examples;temporal-contract"
            TIMEOUT 30
    )
endif()

install(TARGETS contract_max_start_delay
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/temporal_contracts/max_start_delay/cpp"
    COMPONENT holoscan-core-examples
)
