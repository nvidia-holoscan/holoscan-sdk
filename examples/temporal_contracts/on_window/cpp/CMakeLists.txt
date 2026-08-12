# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_contract_on_window LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(contract_on_window contract_on_window.cpp)
target_link_libraries(contract_on_window PRIVATE holoscan::core)
holoscan_configure_example_install(contract_on_window
    INSTALL_RPATH "$ORIGIN/../../../../lib"
)

if(BUILD_TESTING)
    # REQ: SYSTEM-013
    add_test(NAME EXAMPLE_CPP_CONTRACT_ON_WINDOW_TEST COMMAND contract_on_window)
    set_tests_properties(EXAMPLE_CPP_CONTRACT_ON_WINDOW_TEST
        PROPERTIES PASS_REGULAR_EXPRESSION "on_window bracket demo complete"
        LABELS "examples;temporal-contract"
        TIMEOUT 30)
endif()

install(TARGETS contract_on_window
    RUNTIME DESTINATION examples/temporal_contracts/on_window/cpp
    COMPONENT holoscan-examples
)
