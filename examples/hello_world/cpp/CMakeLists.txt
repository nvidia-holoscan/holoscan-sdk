# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_hello_world LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(hello_world hello_world.cpp)
target_link_libraries(hello_world PRIVATE holoscan::core)
holoscan_configure_example_install(hello_world)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_HELLO_WORLD_TEST COMMAND hello_world)
    set_tests_properties(EXAMPLE_CPP_HELLO_WORLD_TEST
        PROPERTIES PASS_REGULAR_EXPRESSION "Hello Holoscan!")
endif()

install(TARGETS hello_world
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/hello_world/cpp"
    COMPONENT holoscan-core-examples
)
