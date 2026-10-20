# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Every case launches a fresh process so environment initialization is observable.
foreach(mode IN ITEMS start compute compute-error advanced cleanup worker)
    foreach(presentation IN ITEMS default on off explicit)
        foreach(environment IN ITEMS unset ON OFF)
            set(command "${CMAKE_COMMAND}" -E env
                --unset=HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING
                HOLOSCAN_LOG_LEVEL=INFO)
            if(NOT environment STREQUAL "unset")
                list(APPEND command "HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING=${environment}")
            endif()
            execute_process(COMMAND ${command} "${PROBE}" "${mode}" "${presentation}"
                RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 10)
            if(NOT result EQUAL 1)
                message(FATAL_ERROR "${mode}/${presentation}/${environment}: exit ${result}: ${output}${error}")
            endif()
            set(automatic TRUE)
            if(environment STREQUAL "OFF" OR (environment STREQUAL "unset" AND
                    (presentation STREQUAL "off" OR presentation STREQUAL "explicit")))
                set(automatic FALSE)
            endif()
            if(automatic AND NOT error MATCHES "RunSession failed:")
                message(FATAL_ERROR "Missing automatic report: ${mode}/${presentation}/${environment}: ${error}")
            elseif(NOT automatic AND error MATCHES "RunSession failed:")
                message(FATAL_ERROR "Unexpected automatic report: ${mode}/${presentation}/${environment}: ${error}")
            endif()
            if((automatic OR presentation STREQUAL "explicit") AND
                    NOT error MATCHES "RGB unsupported; requires RGBA")
                message(FATAL_ERROR "Lost original cause: ${mode}/${presentation}/${environment}: ${error}")
            endif()
            if(NOT mode STREQUAL "advanced" AND NOT error MATCHES "source stopped")
                message(FATAL_ERROR "Cleanup/user output lost: ${mode}/${presentation}/${environment}: ${error}")
            endif()
        endforeach()
    endforeach()
endforeach()
