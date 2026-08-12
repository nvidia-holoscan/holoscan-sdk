# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_benchmarks LANGUAGES CXX)

include(CTest)

find_package(Python3 COMPONENTS Interpreter QUIET)

if(BUILD_TESTING AND Python3_Interpreter_FOUND)
    add_test(
        NAME BENCHMARK_CPU_PERFORMANCE_SCRIPT_TESTS
        COMMAND ${Python3_EXECUTABLE} -m pytest -q
                "${CMAKE_CURRENT_SOURCE_DIR}/test_configure_cpu_performance.py"
    )
    set_tests_properties(BENCHMARK_CPU_PERFORMANCE_SCRIPT_TESTS
        PROPERTIES LABELS "benchmark"
    )
    add_test(
        NAME BENCHMARK_TRANSPORT_SUITE_RUNNER_PYTHON_TESTS
        COMMAND ${Python3_EXECUTABLE} -m pytest -q
                "${CMAKE_CURRENT_SOURCE_DIR}/test_transport_benchmark_suite.py"
    )
    set_tests_properties(BENCHMARK_TRANSPORT_SUITE_RUNNER_PYTHON_TESTS
        PROPERTIES LABELS "benchmark"
    )
endif()

if(NOT TARGET holoscan::core)
    find_package(holoscan REQUIRED CONFIG
        PATHS
            "${CMAKE_CURRENT_LIST_DIR}/.."
            "/opt/nvidia/holoscan"
    )
endif()

if(NOT TARGET benchmark::benchmark_main)
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/setup_dependencies.cmake")
endif()

set(HOLOSCAN_BENCHMARK_RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/run_benchmark.py")

add_custom_target(holoscan-benchmark-targets)
if(HOLOSCAN_BUILD_PYTHON)
    add_custom_target(holoscan-python-benchmark-targets)
endif()

if(CMAKE_CXX_CLANG_TIDY AND NOT "${CMAKE_CXX_CLANG_TIDY}" MATCHES
        "clang-analyzer-deadcode\\.DeadStores")
    # Allow the standard Google Benchmark loop idiom: for (auto _ : state).
    list(TRANSFORM CMAKE_CXX_CLANG_TIDY
         REPLACE "^(--checks=.*)$" "\\1,-clang-analyzer-deadcode.DeadStores")
endif()

add_subdirectory(core_hello)
# The EA1 clock registry is intentionally private. Build its measurement
# harness only in the SDK source tree, where the non-installed implementation
# header is available; installed benchmark packages must remain self-contained.
if(TARGET core)
    add_subdirectory(time)
    add_subdirectory(temporal_contract)
    add_subdirectory(runtime)
    add_subdirectory(graph)
    add_subdirectory(endpoint_catalog)
endif()
if(HOLOSCAN_BUILD_PYTHON)
    add_subdirectory(tensor_python_as_tensor)
endif()
add_subdirectory(channel_pod)
add_subdirectory(channel_schema)
add_subdirectory(channel_tensor)
add_subdirectory(channel_interprocess)
if(TARGET core)
    # The cumulative EA1 suite references the already-defined Runtime and
    # Channel benchmark targets instead of recompiling their implementation.
    add_subdirectory(ea1_runtime)
endif()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    DESTINATION benchmark
    COMPONENT holoscan-benchmarks
    FILES_MATCHING
        PATTERN "CMakeLists.txt"
        PATTERN "README.md"
        PATTERN "*.cmake"
        PATTERN "*.cpp"
        PATTERN "*.hpp"
        PATTERN "*.py"
        PATTERN "*.sh"
    PATTERN "__pycache__" EXCLUDE
)
