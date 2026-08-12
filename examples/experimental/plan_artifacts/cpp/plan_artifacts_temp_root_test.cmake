# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED PLAN_ARTIFACTS_EXAMPLE)
    message(FATAL_ERROR "PLAN_ARTIFACTS_EXAMPLE is required")
endif()

if(NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "TEST_ROOT is required")
endif()

set(unavailable_temp_root "${TEST_ROOT}/plan-artifacts-unavailable-temp-root")
file(REMOVE_RECURSE "${unavailable_temp_root}")
file(WRITE "${unavailable_temp_root}" "not a directory\n")

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "TMPDIR=${unavailable_temp_root}"
        "TMP=${unavailable_temp_root}"
        "TEMP=${unavailable_temp_root}"
        "${PLAN_ARTIFACTS_EXAMPLE}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)

file(REMOVE "${unavailable_temp_root}")

if(result EQUAL 0)
    message(FATAL_ERROR "plan_artifacts unexpectedly accepted an unavailable temporary root")
endif()

set(combined_output "${output}${error}")
if(NOT combined_output MATCHES "failed to create a unique temporary workspace")
    message(FATAL_ERROR
        "plan_artifacts did not report the controlled temporary-root diagnostic:\n"
        "${combined_output}")
endif()

if(combined_output MATCHES "filesystem error|terminate called|uncaught exception")
    message(FATAL_ERROR
        "plan_artifacts allowed a filesystem exception to escape:\n${combined_output}")
endif()

message(STATUS "plan_artifacts unavailable temporary root handled without an exception")
