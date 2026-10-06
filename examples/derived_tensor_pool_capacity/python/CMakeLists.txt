# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_derived_tensor_pool_capacity_python LANGUAGES NONE)
    include(CTest)
    include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/HoloscanExampleInstall.cmake")
    find_package(holoscan REQUIRED CONFIG COMPONENTS config)
    find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
    if(NOT holoscan_PYTHON_PACKAGE_DIR)
        message(FATAL_ERROR "The selected Holoscan SDK package does not provide Python bindings")
    endif()
    set(HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR "${holoscan_PYTHON_PACKAGE_DIR}")
endif()

holoscan_add_python_example(
    NAME derived_tensor_pool_capacity
    SCRIPT derived_tensor_pool_capacity.py
    TEST_NAME EXAMPLE_PYTHON_DERIVED_TENSOR_POOL_CAPACITY_TEST
    LABELS examples python compile tensor parameter
)

if(BUILD_TESTING)
    # Reject a non-RGBA output shape through the installed Python example.
    add_test(NAME EXAMPLE_PYTHON_DERIVED_TENSOR_POOL_CAPACITY_REJECTS_NON_RGBA_TEST
        COMMAND ${CMAKE_COMMAND}
            "-DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
            "-DPYTHONPATH=${HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR}"
            "-DEXAMPLE_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/derived_tensor_pool_capacity.py"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_rejected_shape.cmake"
    )
    set_tests_properties(
        EXAMPLE_PYTHON_DERIVED_TENSOR_POOL_CAPACITY_REJECTS_NON_RGBA_TEST PROPERTIES
        FAIL_REGULAR_EXPRESSION "Fatal Python error"
        LABELS "examples;python;compile;tensor;parameter"
        TIMEOUT 30
    )
endif()
