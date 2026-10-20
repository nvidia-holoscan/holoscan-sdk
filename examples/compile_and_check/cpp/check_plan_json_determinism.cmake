# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED COMPILE_AND_CHECK_EXECUTABLE OR NOT EXISTS "${COMPILE_AND_CHECK_EXECUTABLE}")
    message(FATAL_ERROR "COMPILE_AND_CHECK_EXECUTABLE must name the built example")
endif()
function(_run_compile_and_check output_variable)
    execute_process(
        COMMAND "${COMPILE_AND_CHECK_EXECUTABLE}" --print-plan-json
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr
    )
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR
            "compile_and_check failed with ${_result}\nstdout:\n${_stdout}\nstderr:\n${_stderr}")
    endif()
    if("${_stdout}" STREQUAL "")
        message(FATAL_ERROR "compile_and_check did not print plan.json")
    endif()
    string(JSON _json_type ERROR_VARIABLE _json_error TYPE "${_stdout}")
    if(_json_error OR NOT _json_type STREQUAL "OBJECT")
        message(FATAL_ERROR "compile_and_check did not print a JSON object: ${_json_error}")
    endif()
    set(${output_variable} "${_stdout}" PARENT_SCOPE)
endfunction()

_run_compile_and_check(_first_process_json)
_run_compile_and_check(_second_process_json)

if(NOT "${_first_process_json}" STREQUAL "${_second_process_json}")
    message(FATAL_ERROR "plan.json changed between independent compile_and_check processes")
endif()

string(LENGTH "${_first_process_json}" _json_bytes)
message(STATUS "plan.json is byte-identical across two processes (${_json_bytes} bytes)")
