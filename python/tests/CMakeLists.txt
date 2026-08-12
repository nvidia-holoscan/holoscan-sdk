# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)

set(HOLOSCAN_PYTHON_TEST_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(HOLOSCAN_PYTHON_TEST_BINARY_DIR "${CMAKE_BINARY_DIR}/python/lib/tests")
add_custom_target(holoscan-python-tests
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${HOLOSCAN_PYTHON_TEST_SOURCE_DIR}"
            "${HOLOSCAN_PYTHON_TEST_BINARY_DIR}"
    COMMENT "Copying Holoscan Python tests"
)
add_dependencies(holoscan-python holoscan-python-tests)

include(HoloscanPytestAddTests)

add_subdirectory(core)
add_subdirectory(system)
