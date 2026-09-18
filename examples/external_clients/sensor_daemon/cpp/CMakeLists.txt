# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_sensor_daemon LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(sensor_daemon sensor_daemon.cpp)
target_compile_features(sensor_daemon PRIVATE cxx_std_20)
target_link_libraries(sensor_daemon PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(sensor_daemon)

install(TARGETS sensor_daemon
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/external_clients/sensor_daemon/cpp"
    COMPONENT holoscan-core-examples
)
