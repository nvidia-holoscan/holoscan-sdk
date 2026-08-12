# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_ea1_sensor_daemon LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(ea1_sensor_daemon sensor_daemon.cpp)
target_compile_features(ea1_sensor_daemon PRIVATE cxx_std_20)
target_link_libraries(ea1_sensor_daemon PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(ea1_sensor_daemon
    INSTALL_RPATH "$ORIGIN/../../../../lib"
)

install(TARGETS ea1_sensor_daemon
    RUNTIME DESTINATION examples/ea1_external_clients/sensor_daemon/cpp
    COMPONENT holoscan-examples
)
