# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_camera_imu_control_python LANGUAGES NONE)
    include(CTest)
    include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/HoloscanExampleInstall.cmake")
endif()

holoscan_add_python_example(
    NAME camera_imu_control
    SCRIPT camera_imu_control.py
    TEST_NAME EXAMPLE_PYTHON_CAMERA_IMU_CONTROL_TEST
    # Use separate host-global services when CTest runs the C++ variant in parallel.
    ARGS --topic-prefix ctest.python.camera_imu
    LABELS examples python external-topics graph
)
