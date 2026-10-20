# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS BENCHMARK INVALID_KIND OUTPUT)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

file(REMOVE "${OUTPUT}")
if(INVALID_KIND STREQUAL "topology")
    set(arguments
        --mode compile_scaling
        --topology linear_fiexed
        --operators 1
        --output "${OUTPUT}"
    )
elseif(INVALID_KIND STREQUAL "payload")
    set(arguments
        --mode headroom
        --payload tiny
        --memory host
        --headroom 1/1
        --output "${OUTPUT}"
    )
elseif(INVALID_KIND STREQUAL "lifecycle_zero_operators" OR
       INVALID_KIND STREQUAL "lifecycle_one_operator")
    if(INVALID_KIND STREQUAL "lifecycle_zero_operators")
        set(operator_count 0)
    else()
        set(operator_count 1)
    endif()
    set(arguments
        --mode compile_scaling
        --topology lifecycle_cross_process_rejected
        --operators "${operator_count}"
        --output "${OUTPUT}"
    )
else()
    message(FATAL_ERROR "unknown INVALID_KIND: ${INVALID_KIND}")
endif()

execute_process(
    COMMAND "${BENCHMARK}" ${arguments}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)
if(NOT result STREQUAL "2")
    message(FATAL_ERROR
        "invalid ${INVALID_KIND} did not fail through the CLI parser (exit=${result})\n"
        "stdout:\n${stdout}\nstderr:\n${stderr}"
    )
endif()
if(NOT stderr MATCHES "^usage: rate_planning_benchmark")
    message(FATAL_ERROR
        "invalid ${INVALID_KIND} did not report parser usage\n"
        "stdout:\n${stdout}\nstderr:\n${stderr}"
    )
endif()
if(EXISTS "${OUTPUT}")
    message(FATAL_ERROR "invalid ${INVALID_KIND} created an evidence artifact: ${OUTPUT}")
endif()
