# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_hello_logger LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(hello_logger hello_logger.cpp)
target_compile_features(hello_logger PRIVATE cxx_std_20)
holoscan_apply_logging_policy(hello_logger)
target_link_libraries(hello_logger
    PRIVATE
        holoscan::logger
)
holoscan_configure_example_install(hello_logger)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_HELLO_LOGGER_TEST COMMAND hello_logger)
    set_tests_properties(EXAMPLE_CPP_HELLO_LOGGER_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "holoscan logger sample complete"
    )
endif()

install(TARGETS hello_logger
    RUNTIME DESTINATION examples/hello_logger/cpp
    COMPONENT holoscan-examples
)
