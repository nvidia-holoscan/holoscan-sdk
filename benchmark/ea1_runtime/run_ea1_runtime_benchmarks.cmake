# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(_required IN ITEMS
        HOLOSCAN_EA1_PYTHON
        HOLOSCAN_EA1_BENCHMARK_RUNNER
        HOLOSCAN_EA1_DISPATCHER_BENCHMARK
        HOLOSCAN_EA1_TEMPORAL_BENCHMARK
        HOLOSCAN_EA1_CHANNEL_POD_BENCHMARK
        HOLOSCAN_EA1_RUNTIME_CHANNEL_BENCHMARK)
    if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
        message(FATAL_ERROR "${_required} is required")
    endif()
endforeach()

function(_holoscan_run_ea1_benchmark _label _executable _filter)
    execute_process(
        COMMAND
            "${HOLOSCAN_EA1_PYTHON}"
            "${HOLOSCAN_EA1_BENCHMARK_RUNNER}"
            "${_executable}"
            "--benchmark_filter=${_filter}"
            --benchmark_min_time=0.0001s
        RESULT_VARIABLE _result
    )
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR "EA1 benchmark '${_label}' failed with status ${_result}")
    endif()
endfunction()

# REQ: SYSTEM-005
_holoscan_run_ea1_benchmark(
    "dispatcher-ready-set"
    "${HOLOSCAN_EA1_DISPATCHER_BENCHMARK}"
    "^BM_DispatcherReadySetCycle/(1|8|64)$"
)
_holoscan_run_ea1_benchmark(
    "temporal-selection"
    "${HOLOSCAN_EA1_TEMPORAL_BENCHMARK}"
    "^BM_EA1_S2MultiRate(Fresh|StalledSkip|StalledReuseLast)$"
)
_holoscan_run_ea1_benchmark(
    "raw-channel"
    "${HOLOSCAN_EA1_CHANNEL_POD_BENCHMARK}"
    "^BM_ChannelPOD_OneToOne_Blocking"
)
_holoscan_run_ea1_benchmark(
    "graph-owner-activation"
    "${HOLOSCAN_EA1_RUNTIME_CHANNEL_BENCHMARK}"
    "^BM_ChannelActivationCycleByGraphOwners/2$"
)
