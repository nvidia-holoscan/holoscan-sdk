# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_NOTIFICATION_BRIDGE_UNSUPPORTED_EXECUTABLE)
    message(FATAL_ERROR "unsupported notification bridge test requires an executable")
endif()

execute_process(
    COMMAND "${HOLOSCAN_NOTIFICATION_BRIDGE_UNSUPPORTED_EXECUTABLE}"
    RESULT_VARIABLE unsupported_result
    OUTPUT_VARIABLE unsupported_output
    ERROR_VARIABLE unsupported_error
    TIMEOUT 10
)
if(NOT unsupported_result EQUAL 77)
    message(FATAL_ERROR
        "unsupported notification bridge returned ${unsupported_result}: "
        "${unsupported_output}${unsupported_error}")
endif()

string(STRIP "${unsupported_output}${unsupported_error}" unsupported_log)
string(CONCAT expected_log
    "SKIP: notification_bridge requires Linux eventfd; this platform does not "
    "provide the required waitable-descriptor capability")
if(NOT unsupported_log STREQUAL expected_log)
    message(FATAL_ERROR
        "unsupported notification bridge message mismatch: ${unsupported_log}")
endif()

message(STATUS "${unsupported_log}")
