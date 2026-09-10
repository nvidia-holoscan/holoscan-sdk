# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_derived_tensor_pool_capacity LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(derived_tensor_pool_capacity derived_tensor_pool_capacity.cpp)
target_compile_features(derived_tensor_pool_capacity PRIVATE cxx_std_20)
target_link_libraries(derived_tensor_pool_capacity PRIVATE holoscan::core)
holoscan_configure_example_install(derived_tensor_pool_capacity)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_DERIVED_TENSOR_POOL_CAPACITY_TEST
        COMMAND derived_tensor_pool_capacity)
    set_tests_properties(EXAMPLE_CPP_DERIVED_TENSOR_POOL_CAPACITY_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "derived tensor pool capacity: configured and checked transform bounds verified"
            LABELS "examples;compile;tensor;parameter"
            TIMEOUT 30)
    add_test(NAME EXAMPLE_CPP_DERIVED_TENSOR_POOL_CAPACITY_CUSTOM_SHAPE_TEST
        COMMAND derived_tensor_pool_capacity 720 1280 4)
    set_tests_properties(EXAMPLE_CPP_DERIVED_TENSOR_POOL_CAPACITY_CUSTOM_SHAPE_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "derived tensor pool capacity: configured and checked transform bounds verified"
            LABELS "examples;compile;tensor;parameter"
            TIMEOUT 30)
    add_test(NAME EXAMPLE_CPP_DERIVED_TENSOR_POOL_CAPACITY_REJECTS_NON_RGBA_TEST
        COMMAND derived_tensor_pool_capacity 2 2 3)
    set_tests_properties(EXAMPLE_CPP_DERIVED_TENSOR_POOL_CAPACITY_REJECTS_NON_RGBA_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "CHANNELS must be 4 for the packed RGBA example"
            LABELS "examples;compile;tensor;parameter"
            TIMEOUT 30)
endif()

install(TARGETS derived_tensor_pool_capacity
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/derived_tensor_pool_capacity/cpp"
    COMPONENT holoscan-examples
)
