# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_PARTITION_ENVIRONMENT_PROBE OR
   HOLOSCAN_PARTITION_ENVIRONMENT_PROBE STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_PARTITION_ENVIRONMENT_PROBE is required")
endif()

function(run_partition_environment_case scenario selector)
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
        COMMAND ${environment_command}
                "${HOLOSCAN_PARTITION_ENVIRONMENT_PROBE}" "${scenario}"
        TIMEOUT 60
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR
            "Fresh-process partition environment case '${scenario}' failed (${result})\n"
            "stdout:\n${output}\nstderr:\n${error}")
    endif()
endfunction()

run_partition_environment_case("absent" "absent")
run_partition_environment_case("serial-baseline" "serial_baseline/v1")
run_partition_environment_case("source-chain" "source_chain/v1")
run_partition_environment_case("invalid" "source_chain")
run_partition_environment_case("forbidden-profile" "source_chain/v1")
run_partition_environment_case("worker-baseline" "absent")
run_partition_environment_case("worker-source-chain" "source_chain/v1")
