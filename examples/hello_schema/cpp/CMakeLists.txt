# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_hello_schema LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(hello_schema hello_schema.cpp)
target_compile_features(hello_schema PRIVATE cxx_std_20)
target_link_libraries(hello_schema
    PRIVATE
        holoscan::schema
)
holoscan_configure_example_install(hello_schema)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_HELLO_SCHEMA_TEST COMMAND hello_schema)
    set_tests_properties(EXAMPLE_CPP_HELLO_SCHEMA_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "holoscan schema sample complete"
    )
endif()

install(TARGETS hello_schema
    RUNTIME DESTINATION examples/hello_schema/cpp
    COMPONENT holoscan-examples
)
