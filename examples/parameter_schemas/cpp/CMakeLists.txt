# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_parameter_schemas LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(parameter_schemas parameter_schemas.cpp)
target_compile_features(parameter_schemas PRIVATE cxx_std_20)
target_link_libraries(parameter_schemas PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(parameter_schemas)

if(BUILD_TESTING)
    # REQ: SYSTEM-014
    add_test(NAME EXAMPLE_CPP_PARAMETER_SCHEMAS_TEST COMMAND parameter_schemas)
    set_tests_properties(EXAMPLE_CPP_PARAMETER_SCHEMAS_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "parameter_schemas: accepted=1 rejected=6 typed_validation=kValidatorRejected termination=stopped"
            LABELS "examples;parameter;schema"
            TIMEOUT 30)
endif()

install(TARGETS parameter_schemas
    RUNTIME DESTINATION examples/parameter_schemas/cpp
    COMPONENT holoscan-examples
)
