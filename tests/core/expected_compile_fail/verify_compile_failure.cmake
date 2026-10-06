# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_BUILD_DIR OR NOT DEFINED HOLOSCAN_TARGET OR
   NOT DEFINED HOLOSCAN_EXPECTED_DIAGNOSTIC)
    message(FATAL_ERROR "compile-fail verification requires build directory, target, and diagnostic")
endif()

set(build_command "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}")
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND build_command --config "${CMAKE_BUILD_TYPE}")
endif()
list(APPEND build_command --target "${HOLOSCAN_TARGET}")

execute_process(
    COMMAND ${build_command}
    RESULT_VARIABLE compile_result
    OUTPUT_VARIABLE compile_stdout
    ERROR_VARIABLE compile_stderr
)

if(compile_result EQUAL 0)
    message(FATAL_ERROR "${HOLOSCAN_TARGET} unexpectedly compiled successfully")
endif()

if(NOT HOLOSCAN_EXPECTED_DIAGNOSTIC STREQUAL "HOLOSCAN_ACCEPT_ANY_DIAGNOSTIC")
    set(compile_output "${compile_stdout}\n${compile_stderr}")
    string(FIND "${compile_output}" "${HOLOSCAN_EXPECTED_DIAGNOSTIC}" diagnostic_position)
    if(diagnostic_position EQUAL -1)
        message(FATAL_ERROR
            "${HOLOSCAN_TARGET} failed without the expected diagnostic "
            "'${HOLOSCAN_EXPECTED_DIAGNOSTIC}':\n${compile_output}"
        )
    endif()
endif()
