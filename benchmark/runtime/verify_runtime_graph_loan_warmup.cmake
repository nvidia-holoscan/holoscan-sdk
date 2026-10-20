# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_RUNTIME_GRAPH_BENCHMARK OR
   HOLOSCAN_RUNTIME_GRAPH_BENCHMARK STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_RUNTIME_GRAPH_BENCHMARK is required")
endif()

foreach(delivery IN ITEMS direct holomq)
    set(delivery_arguments)
    if(delivery STREQUAL "holomq")
        list(APPEND delivery_arguments --force-holomq-process-local-inline)
    endif()
    execute_process(
        COMMAND "${HOLOSCAN_RUNTIME_GRAPH_BENCHMARK}"
                --workload latency
                --scenario multi_tensor_loan
                --tensor-storage pageable_host
                --tensor-bytes 64
                --samples 8
                --warmup 8
                --publisher-period-us 1000
                --payload-validation final
                ${delivery_arguments}
                --timeout-ms 10000
        RESULT_VARIABLE benchmark_result
        OUTPUT_VARIABLE benchmark_stdout
        ERROR_VARIABLE benchmark_stderr
    )
    if(NOT benchmark_result EQUAL 0)
        message(FATAL_ERROR
            "${delivery} loan warmup failed:\n${benchmark_stdout}\n${benchmark_stderr}")
    endif()

    string(JSON field_count GET "${benchmark_stdout}" tensor field_count)
    math(EXPR last_field "${field_count} - 1")
    set(expected_initializations 0)
    foreach(field RANGE 0 ${last_field})
        string(JSON slots GET "${benchmark_stdout}" tensor fields ${field} pool_slots)
        math(EXPR expected_initializations "${expected_initializations} + ${slots}")
    endforeach()
    string(JSON initializations GET "${benchmark_stdout}" counts descriptor_backing_initializations)
    string(JSON measured_initializations GET "${benchmark_stdout}"
        counts descriptor_backing_initializations_during_measurement)
    if(NOT initializations EQUAL expected_initializations OR
       NOT measured_initializations EQUAL 0)
        message(FATAL_ERROR
            "${delivery} loan warmup initialized ${initializations}/${expected_initializations} "
            "field-pool backings, including ${measured_initializations} during measurement")
    endif()
endforeach()
