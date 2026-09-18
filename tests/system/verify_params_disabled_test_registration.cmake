# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# REQ: PARAM-024

foreach(required_variable IN ITEMS HOLOSCAN_BUILD_DIR HOLOSCAN_CTEST_COMMAND)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

string(CONCAT params_disabled_compile_fail_pattern
    "^EXPECTED_COMPILE_FAIL[.](parameter_schema_requires_aggregate|"
    "parameter_schema_codec_requires_complete|"
    "parameter_schema_codec_requires_codable_fields)$"
)

set(test_list_command
    "${HOLOSCAN_CTEST_COMMAND}"
    --test-dir "${HOLOSCAN_BUILD_DIR}/tests/core"
    --show-only=json-v1
    -R "${params_disabled_compile_fail_pattern}"
)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND test_list_command -C "${CMAKE_BUILD_TYPE}")
endif()

execute_process(
    COMMAND ${test_list_command}
    RESULT_VARIABLE test_list_result
    OUTPUT_VARIABLE test_list_json
    ERROR_VARIABLE test_list_error
    TIMEOUT 30
)
if(NOT test_list_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to inspect the Params-disabled CTest inventory: ${test_list_error}"
    )
endif()

string(JSON test_count ERROR_VARIABLE test_json_error LENGTH "${test_list_json}" tests)
if(NOT test_json_error STREQUAL "NOTFOUND" OR NOT test_count EQUAL 1)
    message(FATAL_ERROR
        "Params-disabled CTest must retain only the Core schema witness: "
        "${test_list_json}${test_list_error}${test_json_error}"
    )
endif()

string(JSON test_name ERROR_VARIABLE test_name_error GET "${test_list_json}" tests 0 name)
if(NOT test_name_error STREQUAL "NOTFOUND" OR
   NOT test_name STREQUAL "EXPECTED_COMPILE_FAIL.parameter_schema_requires_aggregate")
    message(FATAL_ERROR
        "Unexpected Params-disabled CTest registration: ${test_name}${test_name_error}"
    )
endif()
