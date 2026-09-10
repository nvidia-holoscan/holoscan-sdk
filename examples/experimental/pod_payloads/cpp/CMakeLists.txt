# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_pod_payloads LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(pod_payloads pod_payloads.cpp)
target_compile_features(pod_payloads PRIVATE cxx_std_20)
target_link_libraries(pod_payloads PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(pod_payloads)

if(BUILD_TESTING)
    # REQ: SYSTEM-014
    add_test(NAME EXAMPLE_CPP_POD_PAYLOADS_TEST COMMAND pod_payloads)
    set_tests_properties(EXAMPLE_CPP_POD_PAYLOADS_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "pod_payloads: decoded 6 FusionControl samples \\(last frame=6, size=32 bytes, zero padding\\) termination=stopped"
            LABELS "examples;experimental;pod"
            TIMEOUT 30
    )
endif()

install(TARGETS pod_payloads
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/experimental/pod_payloads/cpp"
    COMPONENT holoscan-examples
)
