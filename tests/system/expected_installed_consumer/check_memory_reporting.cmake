# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Require successful ownership/error assertions as well as the two expected reports. A CTest
# PASS_REGULAR_EXPRESSION alone deliberately ignores the executable's exit status.
execute_process(
    COMMAND "${CONSUMER}" OFF
    RESULT_VARIABLE consumer_result
    OUTPUT_VARIABLE consumer_stdout
    ERROR_VARIABLE consumer_stderr
)
if(NOT consumer_result STREQUAL "0")
    message(FATAL_ERROR "MemoryBuffer consumer failed (${consumer_result}): ${consumer_stderr}")
endif()
foreach(expected_message IN ITEMS "Native allocator failed" "Failed to free memory")
    if(NOT consumer_stderr MATCHES "${expected_message}")
        message(FATAL_ERROR "Missing ${expected_message}: ${consumer_stderr}")
    endif()
endforeach()
