# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_graph_composition_python LANGUAGES NONE)
    include(CTest)
    include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/HoloscanExampleInstall.cmake")
endif()

holoscan_add_python_example(
    NAME graph_composition
    SCRIPT graph_composition.py
    TEST_NAME EXAMPLE_PYTHON_GRAPH_COMPOSITION_TEST
    LABELS examples python graph
)
