# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_ea1_result_recorder LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(ea1_result_recorder recorder.cpp)
target_compile_features(ea1_result_recorder PRIVATE cxx_std_20)
target_link_libraries(ea1_result_recorder PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(ea1_result_recorder
    INSTALL_RPATH "$ORIGIN/../../../../lib"
)

install(TARGETS ea1_result_recorder
    RUNTIME DESTINATION examples/ea1_external_clients/recorder/cpp
    COMPONENT holoscan-examples
)
