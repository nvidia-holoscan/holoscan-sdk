# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_hello_tensor LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(hello_tensor hello_tensor.cpp)
target_compile_features(hello_tensor PRIVATE cxx_std_20)
target_link_libraries(hello_tensor
    PRIVATE
        holoscan::core
)
holoscan_configure_example_install(hello_tensor)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_HELLO_TENSOR_TEST COMMAND hello_tensor)
    set_tests_properties(EXAMPLE_CPP_HELLO_TENSOR_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "holoscan tensor sample complete"
    )
endif()

install(TARGETS hello_tensor
    RUNTIME DESTINATION examples/hello_tensor/cpp
    COMPONENT holoscan-examples
)
