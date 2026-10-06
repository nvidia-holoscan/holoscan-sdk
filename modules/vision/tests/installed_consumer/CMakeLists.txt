# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_vision_installed_consumer LANGUAGES CXX)

include(CTest)

if(HOLOSCAN_PACKAGE_FIND_MODE STREQUAL "component")
    find_package(holoscan COMPONENTS vision REQUIRED)
else()
    if(NOT DEFINED HOLOSCAN_MODULE_PACKAGE_DIR OR HOLOSCAN_MODULE_PACKAGE_DIR STREQUAL "")
        message(FATAL_ERROR "HOLOSCAN_MODULE_PACKAGE_DIR is required for module package discovery")
    endif()
    set(holoscan-vision_DIR "${HOLOSCAN_MODULE_PACKAGE_DIR}")
    find_package(holoscan-vision CONFIG REQUIRED)
endif()

add_executable(holoscan_vision_installed_consumer main.cpp)
target_compile_features(holoscan_vision_installed_consumer PRIVATE cxx_std_20)
target_link_libraries(holoscan_vision_installed_consumer PRIVATE holoscan::vision)

add_test(NAME holoscan_vision_installed_consumer.run
    COMMAND holoscan_vision_installed_consumer
)
