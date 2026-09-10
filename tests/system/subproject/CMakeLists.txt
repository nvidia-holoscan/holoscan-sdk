# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_subproject_configure LANGUAGES C CXX)

if(NOT DEFINED HOLOSCAN_SOURCE_DIR OR HOLOSCAN_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_SOURCE_DIR must name the SDK source directory")
endif()

# Configure every in-tree consumer that uses Holoscan-owned source or build
# paths. This keeps the fixture sensitive to parent-project root leakage in
# benchmarks, examples, Python bindings, and tests.
set(BUILD_TESTING ON CACHE BOOL "" FORCE)
set(HOLOSCAN_BUILD_BENCHMARKS ON CACHE BOOL "" FORCE)
set(HOLOSCAN_BUILD_EXAMPLES ON CACHE BOOL "" FORCE)
set(HOLOSCAN_BUILD_PYTHON ON CACHE BOOL "" FORCE)
set(HOLOSCAN_BUILD_TESTS ON CACHE BOOL "" FORCE)
set(HOLOSCAN_DOWNLOAD_DATASETS OFF CACHE BOOL "" FORCE)
set(HOLOSCAN_USE_SCCACHE OFF CACHE BOOL "" FORCE)

add_subdirectory("${HOLOSCAN_SOURCE_DIR}" holoscan-sdk)

foreach(holoscan_subproject_target IN ITEMS
        channel_test
        core_hello_benchmark
        hello_world
        holoscan_core_python
)
    if(NOT TARGET "${holoscan_subproject_target}")
        message(FATAL_ERROR
            "Expected subproject target '${holoscan_subproject_target}' was not configured"
        )
    endif()
endforeach()

get_target_property(holoscan_core_archive_output_directory core ARCHIVE_OUTPUT_DIRECTORY)
if(NOT holoscan_core_archive_output_directory STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/holoscan-sdk/lib")
    message(FATAL_ERROR
        "Holoscan core archive output directory is '${holoscan_core_archive_output_directory}', "
        "expected '${CMAKE_CURRENT_BINARY_DIR}/holoscan-sdk/lib'"
    )
endif()

get_target_property(
    holoscan_python_library_output_directory
    holoscan_core_python
    LIBRARY_OUTPUT_DIRECTORY
)
set(holoscan_expected_python_library_output_directory
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-sdk/python/lib/holoscan/core"
)
if(NOT holoscan_python_library_output_directory
        STREQUAL holoscan_expected_python_library_output_directory)
    message(FATAL_ERROR
        "Holoscan Python library output directory is "
        "'${holoscan_python_library_output_directory}', expected "
        "'${holoscan_expected_python_library_output_directory}'"
    )
endif()
