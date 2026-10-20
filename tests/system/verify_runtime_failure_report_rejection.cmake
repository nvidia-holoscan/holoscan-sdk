# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

execute_process(
    COMMAND "${CMAKE_COMMAND}" "-DPROBE=${PROBE}" -DMODE=invalid-exit
        -P "${CMAKE_CURRENT_LIST_DIR}/verify_runtime_failure_report.cmake"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE errors
    TIMEOUT 35
)
if(NOT result STREQUAL "1" OR NOT errors MATCHES "Expected failure exit status 1")
    message(FATAL_ERROR "Verifier did not reject unexpected exit status: ${result}\n${output}\n${errors}")
endif()
