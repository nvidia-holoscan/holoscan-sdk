# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_SUBGRAPH_SMOKE_EXECUTABLE OR
   HOLOSCAN_SUBGRAPH_SMOKE_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_SUBGRAPH_SMOKE_EXECUTABLE is required")
endif()

function(run_subgraph_smoke selector output_variable)
    if(selector STREQUAL "absent")
        set(environment_command
            "${CMAKE_COMMAND}" -E env
            --unset=HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM)
    else()
        set(environment_command
            "${CMAKE_COMMAND}" -E env
            "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM=${selector}")
    endif()

    execute_process(
        COMMAND ${environment_command} "${HOLOSCAN_SUBGRAPH_SMOKE_EXECUTABLE}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR
            "Subgraph smoke failed for selector '${selector}' (${result}): ${error}")
    endif()
    if(NOT error STREQUAL "")
        message(FATAL_ERROR
            "Subgraph smoke emitted unexpected stderr for selector '${selector}': ${error}")
    endif()
    set(${output_variable} "${output}" PARENT_SCOPE)
endfunction()

run_subgraph_smoke("absent" absent_output)
foreach(retired_value IN ITEMS "serial_baseline/v1" "source_chain/v1" "invalid-retired-value")
    run_subgraph_smoke("${retired_value}" retired_output)
    if(NOT absent_output STREQUAL retired_output)
        message(FATAL_ERROR
            "Retired selector leaked into Graph authoring for '${retired_value}'\n"
            "absent:\n${absent_output}\npresent:\n${retired_output}")
    endif()
endforeach()
