# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_cupy_pipeline_python LANGUAGES NONE)
    include(CTest)
    include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/HoloscanExampleInstall.cmake")
    find_package(holoscan COMPONENTS config REQUIRED)
    find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
    if(NOT holoscan_PYTHON_PACKAGE_DIR)
        message(FATAL_ERROR "The selected Holoscan SDK package does not provide Python bindings")
    endif()
    set(HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR "${holoscan_PYTHON_PACKAGE_DIR}")
endif()

holoscan_add_python_example(
    NAME cupy_pipeline
    SCRIPT cupy_pipeline.py
    TEST_NAME EXAMPLE_PYTHON_CUPY_PIPELINE_TEST
    ARGS --skip-if-unavailable
    LABELS examples python cuda cupy
)

if(BUILD_TESTING)
    set_tests_properties(EXAMPLE_PYTHON_CUPY_PIPELINE_TEST PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME EXAMPLE_PYTHON_CUPY_PIPELINE_PARTITION_STREAMS_TEST
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR}"
            "PYTHONUNBUFFERED=1"
            "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cupy_pipeline.py"
            --per-partition-streams --skip-if-unavailable
    )
    set_tests_properties(EXAMPLE_PYTHON_CUPY_PIPELINE_PARTITION_STREAMS_TEST PROPERTIES
        FAIL_REGULAR_EXPRESSION "Fatal Python error"
        LABELS "examples;python;cuda;cupy"
        TIMEOUT 30
        SKIP_RETURN_CODE 77
    )
endif()
