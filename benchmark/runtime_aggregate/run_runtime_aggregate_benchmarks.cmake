# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(_required IN ITEMS
        HOLOSCAN_RUNTIME_AGGREGATE_PYTHON
        HOLOSCAN_RUNTIME_AGGREGATE_BENCHMARK_RUNNER
        HOLOSCAN_RUNTIME_AGGREGATE_DISPATCHER_BENCHMARK
        HOLOSCAN_RUNTIME_AGGREGATE_TEMPORAL_BENCHMARK
        HOLOSCAN_RUNTIME_AGGREGATE_CHANNEL_POD_BENCHMARK
        HOLOSCAN_RUNTIME_AGGREGATE_CHANNEL_BENCHMARK)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} is required")
    endif()
endforeach()

function(_holoscan_run_runtime_aggregate_benchmark _label _executable _filter)
    execute_process(
        COMMAND
            "${HOLOSCAN_RUNTIME_AGGREGATE_PYTHON}"
            "${HOLOSCAN_RUNTIME_AGGREGATE_BENCHMARK_RUNNER}"
            "${_executable}"
            "--benchmark_filter=${_filter}"
            --benchmark_min_time=0.0001s
        RESULT_VARIABLE _result
    )
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "Runtime aggregate benchmark '${_label}' failed with status ${_result}")
    endif()
endfunction()

# REQ: SYSTEM-005
_holoscan_run_runtime_aggregate_benchmark(
    "dispatcher-ready-set"
    "${HOLOSCAN_RUNTIME_AGGREGATE_DISPATCHER_BENCHMARK}"
    "^BM_DispatcherReadySetCycle/(1|8|64)$"
)
_holoscan_run_runtime_aggregate_benchmark(
    "temporal-selection"
    "${HOLOSCAN_RUNTIME_AGGREGATE_TEMPORAL_BENCHMARK}"
    "^BM_RuntimeS2MultiRate(Fresh|StalledSkip|StalledReuseLast)$"
)
_holoscan_run_runtime_aggregate_benchmark(
    "raw-channel"
    "${HOLOSCAN_RUNTIME_AGGREGATE_CHANNEL_POD_BENCHMARK}"
    "^BM_ChannelPOD_OneToOne_Blocking"
)
_holoscan_run_runtime_aggregate_benchmark(
    "graph-owner-activation"
    "${HOLOSCAN_RUNTIME_AGGREGATE_CHANNEL_BENCHMARK}"
    "^BM_ChannelActivationCycleByGraphOwners/2$"
)
