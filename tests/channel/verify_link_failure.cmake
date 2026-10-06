# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_BUILD_DIR OR NOT DEFINED HOLOSCAN_TARGET OR
   NOT DEFINED HOLOSCAN_EXPECTED_DIAGNOSTIC)
    message(FATAL_ERROR "link-fail verification requires build directory, target, and diagnostic")
endif()

set(build_command "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}")
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND build_command --config "${CMAKE_BUILD_TYPE}")
endif()
list(APPEND build_command --target "${HOLOSCAN_TARGET}")

execute_process(
    COMMAND ${build_command}
    RESULT_VARIABLE link_result
    OUTPUT_VARIABLE link_stdout
    ERROR_VARIABLE link_stderr
)

if(link_result EQUAL 0)
    message(FATAL_ERROR "${HOLOSCAN_TARGET} unexpectedly linked successfully")
endif()

set(link_output "${link_stdout}\n${link_stderr}")
string(FIND "${link_output}" "${HOLOSCAN_EXPECTED_DIAGNOSTIC}" diagnostic_position)
if(diagnostic_position EQUAL -1)
    message(FATAL_ERROR
        "${HOLOSCAN_TARGET} failed without the expected link diagnostic "
        "'${HOLOSCAN_EXPECTED_DIAGNOSTIC}':\n${link_output}"
    )
endif()
