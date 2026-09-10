# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_RUNTIME_GRAPH_BENCHMARK OR
   HOLOSCAN_RUNTIME_GRAPH_BENCHMARK STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_RUNTIME_GRAPH_BENCHMARK is required")
endif()

execute_process(
    COMMAND "${HOLOSCAN_RUNTIME_GRAPH_BENCHMARK}"
            --workload latency
            --scenario multi_tensor_loan
            --tensor-storage pageable_host
            --tensor-bytes 64
            --samples 8
            --warmup 2
            --publisher-period-us 1000
            --payload-validation final
            --timeout-ms 10000
    RESULT_VARIABLE benchmark_result
    OUTPUT_VARIABLE benchmark_stdout
    ERROR_VARIABLE benchmark_stderr
)

if(benchmark_result EQUAL 0)
    message(FATAL_ERROR "undersized multi-Tensor loan warmup unexpectedly succeeded")
endif()

set(expected_diagnostic "descriptor warmup cannot cover the hot-path Tensor slots")
string(CONCAT benchmark_output "${benchmark_stdout}" "${benchmark_stderr}")
string(FIND "${benchmark_output}" "${expected_diagnostic}" diagnostic_offset)
if(diagnostic_offset EQUAL -1)
    message(FATAL_ERROR
        "undersized multi-Tensor loan warmup omitted the expected diagnostic:\n${benchmark_output}")
endif()
