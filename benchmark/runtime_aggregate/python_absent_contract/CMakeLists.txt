# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(runtime_aggregate_python_absent_contract LANGUAGES CXX)

include(CTest)
set(BUILD_TESTING ON)
set(Python3_Interpreter_FOUND FALSE)

add_library(runtime_aggregate_contract_holoscan_core INTERFACE)
add_library(holoscan::core ALIAS runtime_aggregate_contract_holoscan_core)
add_library(runtime_aggregate_contract_benchmark_main INTERFACE)
add_library(benchmark::benchmark_main ALIAS runtime_aggregate_contract_benchmark_main)
add_custom_target(channel_pod_benchmark)
add_custom_target(runtime_channel_benchmark)
add_custom_target(runtime_dispatcher_benchmark)
add_custom_target(holoscan-benchmark-targets)

add_subdirectory("${HOLOSCAN_RUNTIME_AGGREGATE_SOURCE_DIR}" runtime_aggregate)

if(NOT TARGET runtime_aggregate_benchmark)
    message(FATAL_ERROR "Python-absent configuration removed runtime_aggregate_benchmark")
endif()
get_property(_configured_tests DIRECTORY "${HOLOSCAN_RUNTIME_AGGREGATE_SOURCE_DIR}" PROPERTY TESTS)
if(BENCHMARK_RUNTIME_AGGREGATE IN_LIST _configured_tests)
    message(FATAL_ERROR "Python-absent configuration registered BENCHMARK_RUNTIME_AGGREGATE")
endif()
if(NOT BENCHMARK_RUNTIME_AGGREGATE.python_absent_configuration IN_LIST _configured_tests)
    message(FATAL_ERROR "Python-absent configuration omitted its structural contract test")
endif()
