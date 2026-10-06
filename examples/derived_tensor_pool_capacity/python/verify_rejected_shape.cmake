# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS PYTHON_EXECUTABLE PYTHONPATH EXAMPLE_SCRIPT)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "verify_rejected_shape requires ${required_variable}")
    endif()
endforeach()

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${PYTHONPATH}"
        "PYTHONUNBUFFERED=1"
        "${PYTHON_EXECUTABLE}" "${EXAMPLE_SCRIPT}" 2 2 3
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)

if(NOT result EQUAL 1)
    message(FATAL_ERROR
        "expected rejected shape to exit 1, got ${result}\nstdout:\n${stdout}\nstderr:\n${stderr}")
endif()

set(output "${stdout}\n${stderr}")
if(NOT output MATCHES "CHANNELS must be 4 for the packed RGBA example")
    message(FATAL_ERROR "rejected-shape diagnostic missing:\n${output}")
endif()

message(STATUS "Rejected non-RGBA shape with the expected diagnostic")
