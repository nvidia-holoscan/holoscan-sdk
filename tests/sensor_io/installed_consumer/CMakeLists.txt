# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(sensor_io_installed_consumer LANGUAGES CXX)
find_package(holoscan CONFIG REQUIRED COMPONENTS sensor_io
    PATHS "${CMAKE_PREFIX_PATH}" NO_DEFAULT_PATH)
add_executable(sensor_io_installed_consumer main.cpp)
target_compile_features(sensor_io_installed_consumer PRIVATE cxx_std_20)
target_link_libraries(sensor_io_installed_consumer PRIVATE holoscan::sensor_io)
enable_testing()
add_test(NAME sensor_io_installed_consumer.run COMMAND sensor_io_installed_consumer)
