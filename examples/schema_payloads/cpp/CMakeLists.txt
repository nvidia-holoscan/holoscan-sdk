# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_schema_payloads LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(schema_payloads schema_payloads.cpp)
target_compile_features(schema_payloads PRIVATE cxx_std_20)
target_link_libraries(schema_payloads PRIVATE holoscan::core holoscan::schema Threads::Threads)
holoscan_configure_example_install(schema_payloads)

if(BUILD_TESTING)
    # REQ: SYSTEM-014
    add_test(NAME EXAMPLE_CPP_SCHEMA_PAYLOADS_TEST COMMAND schema_payloads)
    set_tests_properties(EXAMPLE_CPP_SCHEMA_PAYLOADS_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "schema_payloads: strings=5 times=4 imus=4 degraded=1 oversize=kResourceExhausted termination=stopped"
            LABELS "examples;schema"
            TIMEOUT 30)
endif()

install(TARGETS schema_payloads
    RUNTIME DESTINATION examples/schema_payloads/cpp
    COMPONENT holoscan-examples
)
