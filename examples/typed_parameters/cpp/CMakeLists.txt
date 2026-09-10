# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_typed_parameters LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(typed_parameters typed_parameters.cpp)
target_link_libraries(typed_parameters PRIVATE holoscan::core)
holoscan_configure_example_install(typed_parameters)

if(BUILD_TESTING)
    # REQ: SYSTEM-013
    add_test(NAME EXAMPLE_CPP_TYPED_PARAMETERS_TEST COMMAND typed_parameters)
    set_tests_properties(EXAMPLE_CPP_TYPED_PARAMETERS_TEST
        PROPERTIES PASS_REGULAR_EXPRESSION "summary: default_threshold=0\\.5 configured_threshold=0\\.8 configured_accepted=1/2 runtime_error=kInvalidArgument typed_validation=kOutOfRange field=threshold"
        LABELS "examples;parameter"
        TIMEOUT 30)
endif()

install(TARGETS typed_parameters
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/typed_parameters/cpp"
    COMPONENT holoscan-examples
)
