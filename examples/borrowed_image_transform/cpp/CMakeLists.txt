# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_borrowed_image_transform LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(borrowed_image_transform borrowed_image_transform.cpp)
target_compile_features(borrowed_image_transform PRIVATE cxx_std_20)
target_link_libraries(borrowed_image_transform PRIVATE holoscan::sensor_io)
holoscan_configure_example_install(borrowed_image_transform)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_BORROWED_IMAGE_TRANSFORM_TEST COMMAND borrowed_image_transform)
    set_tests_properties(EXAMPLE_CPP_BORROWED_IMAGE_TRANSFORM_TEST PROPERTIES
        PASS_REGULAR_EXPRESSION "borrowed image transform: pixels, metadata, and provenance verified"
        LABELS "examples;sensor-io"
        TIMEOUT 30)
endif()

install(TARGETS borrowed_image_transform
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/borrowed_image_transform/cpp"
    COMPONENT holoscan-core-examples)
