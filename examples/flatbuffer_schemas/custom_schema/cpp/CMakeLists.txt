# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_custom_schema LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/custom_schema_flatbuffers.cmake")

add_executable(custom_schema
    custom_schema.cpp
)
target_compile_features(custom_schema PRIVATE cxx_std_20)
target_link_libraries(custom_schema PRIVATE holoscan::core custom_schema_flatbuffers_generated)
set_target_properties(custom_schema PROPERTIES CXX_EXTENSIONS OFF)
holoscan_configure_example_install(custom_schema)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_CUSTOM_SCHEMA_TEST COMMAND custom_schema)
    set_tests_properties(EXAMPLE_CPP_CUSTOM_SCHEMA_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "custom_schema: received 5 controls"
            LABELS "examples;schema"
            TIMEOUT 30)
endif()

install(TARGETS custom_schema
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/flatbuffer_schemas/custom_schema/cpp"
    COMPONENT holoscan-core-examples
)
install(FILES
    custom_control.fbs
    DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/flatbuffer_schemas/custom_schema/cpp"
    COMPONENT holoscan-core-examples
)
