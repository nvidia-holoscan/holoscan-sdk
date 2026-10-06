# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_hello_core LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(hello_core hello_core.cpp)
target_compile_features(hello_core PRIVATE cxx_std_20)
target_link_libraries(hello_core
    PRIVATE
        holoscan::core
)
holoscan_configure_example_install(hello_core)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_HELLO_CORE_TEST COMMAND hello_core)
    set_tests_properties(EXAMPLE_CPP_HELLO_CORE_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "holoscan core sample complete"
    )
endif()

install(TARGETS hello_core
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/hello_core/cpp"
    COMPONENT holoscan-core-examples
)
