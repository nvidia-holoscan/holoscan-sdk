# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

execute_process(
    COMMAND "${PROBE}" "${MODE}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE errors
    TIMEOUT 30
)
set(report "${output}\n${errors}")
if(MODE STREQUAL "success" OR MODE STREQUAL "advanced-success")
    if(NOT result STREQUAL "0" OR report MATCHES "RunSession failed" OR
       report MATCHES "ignored successful stage detail")
        message(FATAL_ERROR "Successful run reported failure: ${result}\n${report}")
    endif()
else()
    if(NOT result STREQUAL "1")
        message(FATAL_ERROR "Expected failure exit status 1, got ${result}\n${report}")
    endif()
    if(NOT report MATCHES "RGB unsupported; requires RGBA" OR
       NOT report MATCHES "z_display" OR NOT report MATCHES "kInvalidArgument")
        message(FATAL_ERROR "Failure lost original cause or owner: ${result}\n${report}")
    endif()
    string(REGEX MATCHALL "RunSession failed:" summaries "${report}")
    list(LENGTH summaries summary_count)
    if(NOT summary_count EQUAL 1)
        message(FATAL_ERROR "Expected one SDK terminal report: ${report}")
    endif()
    if(report MATCHES "Lifecycle transition failed: operation=0" OR
       report MATCHES "must not replace first reported error")
        message(FATAL_ERROR "Failure reporting lost precedence or emitted numeric-only noise: ${report}")
    endif()
    if(MODE STREQUAL "advanced")
        if(NOT report MATCHES "kConfigure")
            message(FATAL_ERROR "Advanced callback stage was lost: ${report}")
        endif()
    elseif(MODE STREQUAL "compute" OR MODE STREQUAL "compute-error")
        if(NOT report MATCHES "kCompute")
            message(FATAL_ERROR "Compute operation was lost: ${report}")
        endif()
    elseif(NOT report MATCHES "kStart" OR NOT report MATCHES "source stopped")
        message(FATAL_ERROR "Start failure lost its phase or skipped cleanup: ${report}")
    endif()
    if(NOT MODE STREQUAL "advanced")
        string(FIND "${errors}" "source stopped" cleanup_position)
        string(FIND "${errors}" "RunSession failed:" report_position)
        if(cleanup_position LESS 0 OR report_position LESS_EQUAL cleanup_position)
            message(FATAL_ERROR "Terminal report preceded required cleanup: ${report}")
        endif()
    endif()
    if(MODE STREQUAL "cleanup" AND NOT report MATCHES "secondary source cleanup failed")
        message(FATAL_ERROR "Secondary failure was lost: ${report}")
    endif()
    if(MODE STREQUAL "embedded" AND NOT report MATCHES "pipeline/z_display")
        message(FATAL_ERROR "Embedded operator path was lost: ${report}")
    endif()
    if(MODE STREQUAL "async" AND NOT report MATCHES "Session diagnostic:")
        message(FATAL_ERROR "Public diagnostic inspection was not exercised: ${report}")
    endif()
endif()
