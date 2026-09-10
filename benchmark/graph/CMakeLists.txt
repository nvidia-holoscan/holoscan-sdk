# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(graph_authoring_benchmark graph_authoring_benchmark.cpp)
add_dependencies(holoscan-benchmark-targets graph_authoring_benchmark)
target_link_libraries(graph_authoring_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
)

if(BUILD_TESTING)
    add_test(
        NAME BENCHMARK_GRAPH_AUTHORING
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:graph_authoring_benchmark>"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_GRAPH_AUTHORING
        PROPERTIES
            LABELS benchmark
    )
endif()

add_executable(partition_planning_benchmark
    partition_planning_benchmark.cpp
    partition_planning_allocation_probe.cpp
)
add_dependencies(holoscan-benchmark-targets partition_planning_benchmark)
target_include_directories(partition_planning_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
)
target_link_libraries(partition_planning_benchmark
    PRIVATE
        holoscan::core
)

if(BUILD_TESTING)
    add_test(
        NAME BENCHMARK_PARTITION_PLANNING_ALLOCATION_GENERATION
        COMMAND $<TARGET_FILE:partition_planning_benchmark>
                --cross-generation-self-test
    )
    set_tests_properties(BENCHMARK_PARTITION_PLANNING_ALLOCATION_GENERATION
        PROPERTIES LABELS "benchmark;ea2;partitioning"
    )

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME BENCHMARK_PARTITION_PLANNING_RUNNER_TESTS
            COMMAND ${Python3_EXECUTABLE} -m pytest -q
                    "${CMAKE_CURRENT_SOURCE_DIR}/test_run_partition_planning_benchmark.py"
        )
        set_tests_properties(BENCHMARK_PARTITION_PLANNING_RUNNER_TESTS
            PROPERTIES LABELS "benchmark;ea2;partitioning"
        )

        add_test(
            NAME BENCHMARK_PARTITION_PLANNING_SMOKE
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_partition_planning_benchmark.py"
                    "$<TARGET_FILE:partition_planning_benchmark>"
                    --topology linear
                    --topology binary_fanout
                    --operators 0
                    --operators 16
                    --repetitions 2
                    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/partition_planning_smoke"
        )
        set_tests_properties(BENCHMARK_PARTITION_PLANNING_SMOKE
            PROPERTIES
                LABELS "benchmark;ea2;partitioning"
                TIMEOUT 30
        )
    endif()

    foreach(hybrid_operator_count IN ITEMS 1 2 3)
        add_test(
            NAME BENCHMARK_PARTITION_PLANNING_HYBRID_REJECT_${hybrid_operator_count}
            COMMAND ${CMAKE_COMMAND}
                    "-DBENCHMARK=$<TARGET_FILE:partition_planning_benchmark>"
                    "-DOPERATORS=${hybrid_operator_count}"
                    "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/partition_planning_hybrid_${hybrid_operator_count}.json"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/assert_small_hybrid_rejected.cmake"
        )
        set_tests_properties(BENCHMARK_PARTITION_PLANNING_HYBRID_REJECT_${hybrid_operator_count}
            PROPERTIES
                LABELS "benchmark;ea2;partitioning"
        )
    endforeach()
endif()

add_executable(rate_planning_benchmark
    rate_planning_benchmark.cpp
    partition_planning_allocation_probe.cpp
)
add_dependencies(holoscan-benchmark-targets rate_planning_benchmark)
target_include_directories(rate_planning_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
)
target_link_libraries(rate_planning_benchmark
    PRIVATE
        holoscan::core
)

if(BUILD_TESTING)
    foreach(invalid_kind IN ITEMS
            topology
            payload
            lifecycle_zero_operators
            lifecycle_one_operator)
        string(TOUPPER "${invalid_kind}" invalid_kind_upper)
        add_test(
            NAME BENCHMARK_RATE_PLANNING_REJECTS_INVALID_${invalid_kind_upper}
            COMMAND ${CMAKE_COMMAND}
                    "-DBENCHMARK=$<TARGET_FILE:rate_planning_benchmark>"
                    "-DINVALID_KIND=${invalid_kind}"
                    "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/invalid_${invalid_kind}.json"
                    -P
                    "${CMAKE_CURRENT_SOURCE_DIR}/assert_rate_planning_invalid_options_rejected.cmake"
        )
        set_tests_properties(BENCHMARK_RATE_PLANNING_REJECTS_INVALID_${invalid_kind_upper}
            PROPERTIES LABELS "benchmark;ea2;rate"
        )
    endforeach()
endif()

if(BUILD_TESTING AND Python3_Interpreter_FOUND)
    add_test(
        NAME BENCHMARK_RATE_PLANNING_RUNNER_TESTS
        COMMAND ${Python3_EXECUTABLE} -m pytest -q
                "${CMAKE_CURRENT_SOURCE_DIR}/test_run_rate_planning_benchmark.py"
    )
    set_tests_properties(BENCHMARK_RATE_PLANNING_RUNNER_TESTS
        PROPERTIES LABELS "benchmark;ea2;rate"
    )

    add_test(
        NAME BENCHMARK_RATE_PLANNING_SMOKE
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_rate_planning_benchmark.py"
                "$<TARGET_FILE:rate_planning_benchmark>"
                --mode compile_scaling
                --topology dual_rate_fusion
                --topology lifecycle_cross_process_rejected
                --operators 16
                --mode headroom
                --payload small
                --memory host
                --headroom 1/1
                --headroom 2/1
                --repetitions 1
                --output-dir "${CMAKE_CURRENT_BINARY_DIR}/rate_planning_smoke"
    )
    set_tests_properties(BENCHMARK_RATE_PLANNING_SMOKE
        PROPERTIES
            LABELS "benchmark;ea2;rate"
            TIMEOUT 60
    )

    add_test(
        NAME BENCHMARK_RATE_RESTART_CONFORMANCE_RUNNER_TESTS
        COMMAND ${Python3_EXECUTABLE} -m pytest -q
                "${CMAKE_CURRENT_SOURCE_DIR}/test_run_rate_restart_conformance_probe.py"
    )
    set_tests_properties(BENCHMARK_RATE_RESTART_CONFORMANCE_RUNNER_TESTS
        PROPERTIES LABELS "benchmark;ea2;rate;runtime"
    )

    if(HOLOSCAN_BUILD_TESTS)
        add_test(
            NAME BENCHMARK_RATE_RESTART_CONFORMANCE_SMOKE
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_rate_restart_conformance_probe.py"
                    "$<TARGET_FILE:runtime_test>"
                    --repetitions 3
                    --process-timeout-sec 15
                    --source-state cmake-smoke-build
                    --output-dir
                    "${CMAKE_CURRENT_BINARY_DIR}/rate_restart_conformance_smoke"
        )
        set_tests_properties(BENCHMARK_RATE_RESTART_CONFORMANCE_SMOKE
            PROPERTIES
                LABELS "benchmark;ea2;rate;runtime"
                # Six isolated scenario processes per repetition, each with a 15-second bound.
                TIMEOUT 300
        )
    endif()
endif()
