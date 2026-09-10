# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_result_recorder LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(result_recorder recorder.cpp)
target_compile_features(result_recorder PRIVATE cxx_std_20)
target_link_libraries(result_recorder PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(result_recorder)

install(TARGETS result_recorder
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/external_clients/recorder/cpp"
    COMPONENT holoscan-examples
)
