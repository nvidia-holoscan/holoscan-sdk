# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_rate_aware_ports LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

find_package(Threads REQUIRED)

add_executable(rate_aware_ports rate_aware_ports.cpp)
target_compile_features(rate_aware_ports PRIVATE cxx_std_20)
target_link_libraries(rate_aware_ports PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(rate_aware_ports)

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_RATE_AWARE_PORTS_TEST COMMAND rate_aware_ports)
    set_tests_properties(EXAMPLE_CPP_RATE_AWARE_PORTS_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "rate_aware_ports example complete"
            LABELS "examples;compile;rate;lifecycle"
            TIMEOUT 30)

    foreach(mode IN ITEMS declarations fusion fallback discover late-binding restart)
        string(TOUPPER "${mode}" mode_upper)
        string(REPLACE "-" "_" mode_upper "${mode_upper}")
        add_test(NAME "EXAMPLE_CPP_RATE_AWARE_PORTS_${mode_upper}_TEST"
            COMMAND rate_aware_ports --mode "${mode}")
        set_tests_properties("EXAMPLE_CPP_RATE_AWARE_PORTS_${mode_upper}_TEST"
            PROPERTIES
                PASS_REGULAR_EXPRESSION "rate_aware_ports ${mode}: PASS"
                LABELS "examples;compile;rate;lifecycle"
                TIMEOUT 30)
    endforeach()

    foreach(missing_record IN ITEMS provider pool)
        string(TOUPPER "${missing_record}" missing_record_upper)
        add_test(
            NAME "EXAMPLE_CPP_RATE_AWARE_PORTS_MISSING_${missing_record_upper}_EVIDENCE_TEST"
            COMMAND "${CMAKE_COMMAND}"
                "-DRATE_AWARE_PORTS_EXECUTABLE=$<TARGET_FILE:rate_aware_ports>"
                "-DMISSING_RECORD=${missing_record}"
                -P "${CMAKE_CURRENT_LIST_DIR}/assert_restart_evidence_rejected.cmake")
        set_tests_properties(
            "EXAMPLE_CPP_RATE_AWARE_PORTS_MISSING_${missing_record_upper}_EVIDENCE_TEST"
            PROPERTIES
                LABELS "examples;compile;rate;lifecycle"
                TIMEOUT 30)
    endforeach()
endif()

install(TARGETS rate_aware_ports
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/rate_aware_ports/cpp"
    COMPONENT holoscan-core-examples
)
