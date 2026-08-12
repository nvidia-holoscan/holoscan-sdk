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
run_subgraph_smoke("serial_baseline/v1" baseline_output)
run_subgraph_smoke("source_chain/v1" source_chain_output)

if(NOT absent_output STREQUAL baseline_output)
    message(FATAL_ERROR
        "Subgraph topology/provenance changed under serial_baseline/v1\n"
        "absent:\n${absent_output}\nserial_baseline/v1:\n${baseline_output}")
endif()
if(NOT absent_output STREQUAL source_chain_output)
    message(FATAL_ERROR
        "Subgraph topology/provenance changed under source_chain/v1\n"
        "absent:\n${absent_output}\nsource_chain/v1:\n${source_chain_output}")
endif()
