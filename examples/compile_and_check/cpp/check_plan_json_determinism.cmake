# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED COMPILE_AND_CHECK_EXECUTABLE OR NOT EXISTS "${COMPILE_AND_CHECK_EXECUTABLE}")
    message(FATAL_ERROR "COMPILE_AND_CHECK_EXECUTABLE must name the built example")
endif()
if(NOT DEFINED DETERMINISM_WORKING_DIRECTORY)
    message(FATAL_ERROR "DETERMINISM_WORKING_DIRECTORY must be set")
endif()

file(MAKE_DIRECTORY "${DETERMINISM_WORKING_DIRECTORY}")
set(_plan_json "${DETERMINISM_WORKING_DIRECTORY}/compile_and_check_plan.json")

function(_run_compile_and_check output_variable)
    file(REMOVE "${_plan_json}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
                "TMPDIR=${DETERMINISM_WORKING_DIRECTORY}"
                "${COMPILE_AND_CHECK_EXECUTABLE}"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr
    )
    if(NOT _result EQUAL 0)
        message(FATAL_ERROR
            "compile_and_check failed with ${_result}\nstdout:\n${_stdout}\nstderr:\n${_stderr}")
    endif()
    if(NOT EXISTS "${_plan_json}")
        message(FATAL_ERROR "compile_and_check did not write ${_plan_json}")
    endif()
    file(READ "${_plan_json}" _json)
    set(${output_variable} "${_json}" PARENT_SCOPE)
endfunction()

_run_compile_and_check(_first_process_json)
_run_compile_and_check(_second_process_json)

if(NOT "${_first_process_json}" STREQUAL "${_second_process_json}")
    message(FATAL_ERROR "plan.json changed between independent compile_and_check processes")
endif()

string(LENGTH "${_first_process_json}" _json_bytes)
message(STATUS "plan.json is byte-identical across two processes (${_json_bytes} bytes)")
