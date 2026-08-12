# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED PROGRAM OR PROGRAM STREQUAL "")
    message(FATAL_ERROR "PROGRAM must name the contract_deadlines executable")
endif()

execute_process(
    COMMAND "${PROGRAM}" --fail-run
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)

if("${result}" STREQUAL "0")
    message(FATAL_ERROR
        "contract_deadlines --fail-run unexpectedly exited 0\n"
        "stdout:\n${stdout}\n"
        "stderr:\n${stderr}")
endif()

set(combined "${stdout}${stderr}")
string(REPLACE "\r\n" "\n" combined "${combined}")
set(framed_output "\n${combined}\n")
string(FIND
    "${framed_output}"
    "\ncontract deadlines fail-run complete\n"
    token_position
)
if(token_position EQUAL -1)
    message(FATAL_ERROR
        "contract_deadlines --fail-run did not print the exact completion token\n"
        "exit: ${result}\n"
        "stdout:\n${stdout}\n"
        "stderr:\n${stderr}")
endif()
