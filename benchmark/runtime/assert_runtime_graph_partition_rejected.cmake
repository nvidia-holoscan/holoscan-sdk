# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_RUNTIME_GRAPH_BENCHMARK OR
   HOLOSCAN_RUNTIME_GRAPH_BENCHMARK STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_RUNTIME_GRAPH_BENCHMARK is required")
endif()
if(NOT DEFINED REJECTION_CASE)
    message(FATAL_ERROR "REJECTION_CASE is required")
endif()

if(REJECTION_CASE STREQUAL "automatic_shared")
    set(rejection_arguments --partition-cover automatic --operator-lanes shared)
    set(expected_diagnostic
        "automatic partition-cover rows require --operator-lanes separate")
elseif(REJECTION_CASE STREQUAL "automatic_forced_holomq")
    set(rejection_arguments
        --partition-cover automatic
        --operator-lanes separate
        --force-holomq-process-local-inline)
    set(expected_diagnostic
        "automatic partition-cover rows cannot force HoloMQ because compiler-derived ownership")
else()
    message(FATAL_ERROR "unknown REJECTION_CASE: ${REJECTION_CASE}")
endif()

execute_process(
    COMMAND "${HOLOSCAN_RUNTIME_GRAPH_BENCHMARK}"
            ${rejection_arguments}
            --output "${CMAKE_CURRENT_BINARY_DIR}/${REJECTION_CASE}.json"
    RESULT_VARIABLE benchmark_result
    OUTPUT_VARIABLE benchmark_stdout
    ERROR_VARIABLE benchmark_stderr
    TIMEOUT 10
)

if(benchmark_result MATCHES "timeout")
    message(FATAL_ERROR "${REJECTION_CASE} timed out")
endif()
if(benchmark_result EQUAL 0)
    message(FATAL_ERROR "${REJECTION_CASE} unexpectedly succeeded")
endif()

string(CONCAT benchmark_output "${benchmark_stdout}" "${benchmark_stderr}")
string(FIND "${benchmark_output}" "${expected_diagnostic}" diagnostic_offset)
if(diagnostic_offset EQUAL -1)
    message(FATAL_ERROR
        "${REJECTION_CASE} omitted the expected diagnostic:\n${benchmark_output}")
endif()
