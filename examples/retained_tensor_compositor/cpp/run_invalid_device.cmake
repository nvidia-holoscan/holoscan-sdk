# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_RETAINED_TENSOR_COMPOSITOR_EXECUTABLE OR
   HOLOSCAN_RETAINED_TENSOR_COMPOSITOR_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_RETAINED_TENSOR_COMPOSITOR_EXECUTABLE is required")
endif()

set(invalid_device 2147483647)
execute_process(
    COMMAND
        "${HOLOSCAN_RETAINED_TENSOR_COMPOSITOR_EXECUTABLE}"
        --device
        "0invalid"
    RESULT_VARIABLE invalid_syntax_result
    OUTPUT_VARIABLE invalid_syntax_output
    ERROR_VARIABLE invalid_syntax_error
    TIMEOUT 30
)
string(CONCAT invalid_syntax_log "${invalid_syntax_output}" "${invalid_syntax_error}")
if(NOT invalid_syntax_result EQUAL 2 OR
   NOT invalid_syntax_log MATCHES
       "usage: retained_tensor_compositor \\[--cuda \\[--device <ordinal>\\]\\]")
    message(FATAL_ERROR
        "Device ordinal with trailing characters was not rejected exactly: "
        "${invalid_syntax_log}")
endif()

execute_process(
    COMMAND
        "${HOLOSCAN_RETAINED_TENSOR_COMPOSITOR_EXECUTABLE}"
        --cuda
        --device
        "${invalid_device}"
    RESULT_VARIABLE compositor_result
    OUTPUT_VARIABLE compositor_output
    ERROR_VARIABLE compositor_error
    TIMEOUT 30
)
string(CONCAT compositor_log "${compositor_output}" "${compositor_error}")

if(compositor_result EQUAL 77)
    if(NOT compositor_log MATCHES
       "retained_tensor_compositor: skipped: CUDA device unavailable")
        message(FATAL_ERROR
            "CUDA-unavailable path did not report its exact skip message: ${compositor_log}")
    endif()
    message(STATUS "${compositor_log}")
    cmake_language(EXIT 77)
endif()

if(NOT compositor_result EQUAL 2)
    message(FATAL_ERROR
        "Invalid authored CUDA device returned ${compositor_result}, expected 2: ${compositor_log}")
endif()
if(NOT compositor_log MATCHES
   "authored CUDA device ordinal ${invalid_device} is outside \\[0, [0-9]+\\)")
    message(FATAL_ERROR
        "Invalid authored CUDA device did not report the precise range error: ${compositor_log}")
endif()
message(STATUS "${compositor_log}")
