# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(unit)
# Layer-API and backend tests exercise the operator library's object code; they live beside it
# but are wired from here so add_test() runs after enable_testing() in top-level module builds.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../operators/tests/unit" operators_unit)

if(NOT holoscan_visualization_IS_TOP_LEVEL)
    add_subdirectory(system)
    # Operator-level HolovizOp tests need Holoscan Core, so they only exist in the SDK build.
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../operators/tests/system" operators_system)
endif()
