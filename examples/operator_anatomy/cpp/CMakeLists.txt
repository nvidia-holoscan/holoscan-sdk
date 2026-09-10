# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_operator_anatomy LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(operator_anatomy operator_anatomy.cpp)
target_compile_features(operator_anatomy PRIVATE cxx_std_20)
target_link_libraries(operator_anatomy PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(operator_anatomy)

if(BUILD_TESTING)
    # REQ: SYSTEM-014
    add_test(NAME EXAMPLE_CPP_OPERATOR_ANATOMY_TEST COMMAND operator_anatomy)
    set_tests_properties(EXAMPLE_CPP_OPERATOR_ANATOMY_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "operator_anatomy complete: accessors verified termination=stopped"
            LABELS "examples;authoring"
            TIMEOUT 30)
endif()

install(TARGETS operator_anatomy
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/operator_anatomy/cpp"
    COMPONENT holoscan-examples
)
