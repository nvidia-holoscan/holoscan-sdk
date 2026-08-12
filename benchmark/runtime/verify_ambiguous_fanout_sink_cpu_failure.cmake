# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_RUNTIME_GRAPH_BENCHMARK OR
   HOLOSCAN_RUNTIME_GRAPH_BENCHMARK STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_RUNTIME_GRAPH_BENCHMARK is required")
endif()

execute_process(
    COMMAND "${HOLOSCAN_RUNTIME_GRAPH_BENCHMARK}"
            --scenario pod
            --operator-lanes separate
            --fanout 2
            --sink-cpu 3
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)

if(result EQUAL 0)
    message(FATAL_ERROR "ambiguous fanout sink CPU was unexpectedly accepted")
endif()

set(expected
    "--sink-cpu with fanout greater than one requires --operator-lanes per_sink and --sink-cpus")
set(output "${stdout}\n${stderr}")
string(FIND "${output}" "${expected}" expected_offset)
if(expected_offset EQUAL -1)
    message(FATAL_ERROR
        "ambiguous fanout sink CPU failed without the required diagnostic:\n${output}")
endif()
