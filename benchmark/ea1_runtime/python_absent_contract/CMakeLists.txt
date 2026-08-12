# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(ea1_runtime_python_absent_contract LANGUAGES CXX)

include(CTest)
set(BUILD_TESTING ON)
set(Python3_Interpreter_FOUND FALSE)

add_library(ea1_contract_holoscan_core INTERFACE)
add_library(holoscan::core ALIAS ea1_contract_holoscan_core)
add_library(ea1_contract_benchmark_main INTERFACE)
add_library(benchmark::benchmark_main ALIAS ea1_contract_benchmark_main)
add_custom_target(channel_pod_benchmark)
add_custom_target(runtime_channel_benchmark)
add_custom_target(runtime_dispatcher_benchmark)
add_custom_target(holoscan-benchmark-targets)

add_subdirectory("${HOLOSCAN_EA1_RUNTIME_SOURCE_DIR}" ea1_runtime)

if(NOT TARGET ea1_runtime_benchmark)
    message(FATAL_ERROR "Python-absent configuration removed ea1_runtime_benchmark")
endif()
get_property(_configured_tests DIRECTORY "${HOLOSCAN_EA1_RUNTIME_SOURCE_DIR}" PROPERTY TESTS)
if(BENCHMARK_EA1_RUNTIME IN_LIST _configured_tests)
    message(FATAL_ERROR "Python-absent configuration registered BENCHMARK_EA1_RUNTIME")
endif()
if(NOT BENCHMARK_EA1_RUNTIME.python_absent_configuration IN_LIST _configured_tests)
    message(FATAL_ERROR "Python-absent configuration omitted its structural contract test")
endif()
