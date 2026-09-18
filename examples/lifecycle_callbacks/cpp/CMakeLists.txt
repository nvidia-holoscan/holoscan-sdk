# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_lifecycle_callbacks LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(lifecycle_callbacks lifecycle_callbacks.cpp)
target_compile_features(lifecycle_callbacks PRIVATE cxx_std_20)
target_link_libraries(lifecycle_callbacks PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(lifecycle_callbacks)

if(BUILD_TESTING)
    # REQ: SYSTEM-014
    add_test(NAME EXAMPLE_CPP_LIFECYCLE_CALLBACKS_TEST COMMAND lifecycle_callbacks)
    set_tests_properties(EXAMPLE_CPP_LIFECYCLE_CALLBACKS_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "lifecycle_callbacks: stage arc verified \\(8 events; producer precedes consumer at allocate and start\\) termination=stopped"
            LABELS "examples;lifecycle"
            TIMEOUT 30)
endif()

install(TARGETS lifecycle_callbacks
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/lifecycle_callbacks/cpp"
    COMPONENT holoscan-core-examples
)
