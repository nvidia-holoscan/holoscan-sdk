# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS BENCHMARK OPERATORS OUTPUT)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

execute_process(
    COMMAND "${BENCHMARK}"
            --topology hybrid
            --operators "${OPERATORS}"
            --output "${OUTPUT}"
    RESULT_VARIABLE benchmark_result
    OUTPUT_VARIABLE benchmark_stdout
    ERROR_VARIABLE benchmark_stderr
)

if(NOT benchmark_result EQUAL 2)
    message(FATAL_ERROR
        "small hybrid benchmark returned ${benchmark_result}, expected 2:\n"
        "${benchmark_stdout}${benchmark_stderr}")
endif()

set(expected_diagnostic "hybrid topology requires zero or at least four operators")
string(CONCAT benchmark_output "${benchmark_stdout}" "${benchmark_stderr}")
string(FIND "${benchmark_output}" "${expected_diagnostic}" diagnostic_offset)
if(diagnostic_offset EQUAL -1)
    message(FATAL_ERROR
        "small hybrid benchmark omitted the expected admission diagnostic:\n${benchmark_output}")
endif()
