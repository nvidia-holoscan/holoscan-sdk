# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_running_and_stopping_python LANGUAGES NONE)
    include(CTest)
    include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/HoloscanExampleInstall.cmake")
endif()

holoscan_add_python_example(
    NAME running_and_stopping
    SCRIPT running_and_stopping.py
    TEST_NAME EXAMPLE_PYTHON_RUNNING_AND_STOPPING_TEST
    LABELS examples python run-session
)
