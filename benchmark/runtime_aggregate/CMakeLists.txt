# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(temporal_benchmark runtime_aggregate_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets temporal_benchmark)
target_include_directories(temporal_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
)
target_link_libraries(temporal_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
)

# Preserve one convenient cumulative build target without compiling a second
# copy of production Runtime or the lower-level benchmark translation units.
add_custom_target(runtime_aggregate_benchmark
    DEPENDS
        channel_pod_benchmark
        temporal_benchmark
        runtime_channel_benchmark
        runtime_dispatcher_benchmark
        runtime_graph_benchmark
        runtime_graph_interprocess_benchmark
)
add_dependencies(holoscan-benchmark-targets runtime_aggregate_benchmark)

if(BUILD_TESTING)
    # REQ: SYSTEM-005
    add_test(
        NAME BENCHMARK_RUNTIME_AGGREGATE.python_absent_configuration
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_RUNTIME_AGGREGATE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DHOLOSCAN_RUNTIME_AGGREGATE_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/python_absent_contract"
            "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
            "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/run_python_absent_contract.cmake"
    )
    set_tests_properties(BENCHMARK_RUNTIME_AGGREGATE.python_absent_configuration
        PROPERTIES
            LABELS "benchmark;system;configuration"
            RUN_SERIAL TRUE
    )
endif()

if(BUILD_TESTING AND Python3_Interpreter_FOUND)
    # REQ: SYSTEM-005
    add_test(
        NAME BENCHMARK_RUNTIME_AGGREGATE
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_RUNTIME_AGGREGATE_PYTHON=${Python3_EXECUTABLE}"
            "-DHOLOSCAN_RUNTIME_AGGREGATE_BENCHMARK_RUNNER=${HOLOSCAN_BENCHMARK_RUNNER}"
            "-DHOLOSCAN_RUNTIME_AGGREGATE_DISPATCHER_BENCHMARK=$<TARGET_FILE:runtime_dispatcher_benchmark>"
            "-DHOLOSCAN_RUNTIME_AGGREGATE_TEMPORAL_BENCHMARK=$<TARGET_FILE:temporal_benchmark>"
            "-DHOLOSCAN_RUNTIME_AGGREGATE_CHANNEL_POD_BENCHMARK=$<TARGET_FILE:channel_pod_benchmark>"
            "-DHOLOSCAN_RUNTIME_AGGREGATE_CHANNEL_BENCHMARK=$<TARGET_FILE:runtime_channel_benchmark>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_aggregate_benchmarks.cmake"
    )
    set_tests_properties(BENCHMARK_RUNTIME_AGGREGATE
        PROPERTIES
            LABELS "benchmark;system"
            TIMEOUT 120
    )
endif()
