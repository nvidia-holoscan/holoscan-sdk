# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_vision_installed_consumer LANGUAGES CXX)

include(CTest)

find_package(holoscan-vision CONFIG REQUIRED)

add_executable(holoscan_vision_installed_consumer main.cpp)
target_compile_features(holoscan_vision_installed_consumer PRIVATE cxx_std_20)
target_link_libraries(holoscan_vision_installed_consumer PRIVATE holoscan::vision)

add_test(NAME holoscan_vision_installed_consumer.run
    COMMAND holoscan_vision_installed_consumer
)
