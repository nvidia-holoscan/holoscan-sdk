# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(ea1_temporal_benchmark ea1_runtime_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets ea1_temporal_benchmark)
target_include_directories(ea1_temporal_benchmark
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(ea1_temporal_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
)

# Preserve one convenient cumulative build target without compiling a second
# copy of production Runtime or the lower-level benchmark translation units.
add_custom_target(ea1_runtime_benchmark
    DEPENDS
        channel_pod_benchmark
        ea1_temporal_benchmark
        runtime_channel_benchmark
        runtime_dispatcher_benchmark
        runtime_graph_benchmark
        runtime_graph_interprocess_benchmark
)
add_dependencies(holoscan-benchmark-targets ea1_runtime_benchmark)

if(BUILD_TESTING)
    # REQ: SYSTEM-005
    add_test(
        NAME BENCHMARK_EA1_RUNTIME.python_absent_configuration
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_EA1_RUNTIME_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DHOLOSCAN_EA1_RUNTIME_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/python_absent_contract"
            "-DHOLOSCAN_GENERATOR=${CMAKE_GENERATOR}"
            "-DHOLOSCAN_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DHOLOSCAN_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DHOLOSCAN_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DHOLOSCAN_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/run_python_absent_contract.cmake"
    )
    set_tests_properties(BENCHMARK_EA1_RUNTIME.python_absent_configuration
        PROPERTIES
            LABELS "benchmark;ea1;system;configuration"
            RUN_SERIAL TRUE
    )
endif()

if(BUILD_TESTING AND Python3_Interpreter_FOUND)
    # REQ: SYSTEM-005
    add_test(
        NAME BENCHMARK_EA1_RUNTIME
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_EA1_PYTHON=${Python3_EXECUTABLE}"
            "-DHOLOSCAN_EA1_BENCHMARK_RUNNER=${HOLOSCAN_BENCHMARK_RUNNER}"
            "-DHOLOSCAN_EA1_DISPATCHER_BENCHMARK=$<TARGET_FILE:runtime_dispatcher_benchmark>"
            "-DHOLOSCAN_EA1_TEMPORAL_BENCHMARK=$<TARGET_FILE:ea1_temporal_benchmark>"
            "-DHOLOSCAN_EA1_CHANNEL_POD_BENCHMARK=$<TARGET_FILE:channel_pod_benchmark>"
            "-DHOLOSCAN_EA1_RUNTIME_CHANNEL_BENCHMARK=$<TARGET_FILE:runtime_channel_benchmark>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/run_ea1_runtime_benchmarks.cmake"
    )
    set_tests_properties(BENCHMARK_EA1_RUNTIME
        PROPERTIES
            LABELS "benchmark;ea1;system"
            TIMEOUT 120
    )
endif()
