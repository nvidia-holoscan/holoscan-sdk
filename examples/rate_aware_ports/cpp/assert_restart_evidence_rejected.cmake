# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED RATE_AWARE_PORTS_EXECUTABLE OR NOT DEFINED MISSING_RECORD)
    message(FATAL_ERROR "RATE_AWARE_PORTS_EXECUTABLE and MISSING_RECORD are required")
endif()

execute_process(
    COMMAND "${RATE_AWARE_PORTS_EXECUTABLE}"
            --self-check-restart-evidence "${MISSING_RECORD}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)

string(CONCAT combined "${output}" "${error}")
if(NOT result EQUAL 2)
    message(FATAL_ERROR
        "missing ${MISSING_RECORD} evidence returned ${result}; expected 2\n${combined}")
endif()
if(NOT combined MATCHES
       "rate_aware_ports restart-evidence-self-check: REJECTED missing=${MISSING_RECORD}")
    message(FATAL_ERROR
        "missing ${MISSING_RECORD} evidence did not produce the guarded diagnostic\n${combined}")
endif()
