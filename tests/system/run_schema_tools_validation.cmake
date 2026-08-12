# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_SCHEMA_TOOLS_MODULE
        HOLOSCAN_SCHEMA_TEST_SOURCE_DIR
        HOLOSCAN_SCHEMA_TEST_WORK_DIR
        HOLOSCAN_FLATC_EXECUTABLE
        HOLOSCAN_FLATC_VERSION
        HOLOSCAN_FLATBUFFERS_REFLECTION_SCHEMA)
    if(NOT DEFINED "${required_variable}" OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

function(run_schema_tools_case case_name expected_diagnostic)
    set(case_build_dir "${HOLOSCAN_SCHEMA_TEST_WORK_DIR}/${case_name}")
    file(REMOVE_RECURSE "${case_build_dir}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            -S "${HOLOSCAN_SCHEMA_TEST_SOURCE_DIR}"
            -B "${case_build_dir}"
            "-DHOLOSCAN_SCHEMA_TOOLS_MODULE=${HOLOSCAN_SCHEMA_TOOLS_MODULE}"
            "-DHOLOSCAN_FLATC_EXECUTABLE=${HOLOSCAN_FLATC_EXECUTABLE}"
            "-DHOLOSCAN_FLATC_VERSION=${HOLOSCAN_FLATC_VERSION}"
            "-DHOLOSCAN_FLATBUFFERS_REFLECTION_SCHEMA=${HOLOSCAN_FLATBUFFERS_REFLECTION_SCHEMA}"
            "-DHOLOSCAN_SCHEMA_TEST_CASE=${case_name}"
        RESULT_VARIABLE configure_result
        OUTPUT_VARIABLE configure_stdout
        ERROR_VARIABLE configure_stderr
    )
    set(configure_output "${configure_stdout}${configure_stderr}")
    if(expected_diagnostic STREQUAL "")
        if(NOT configure_result EQUAL 0)
            message(FATAL_ERROR
                "Schema-tools success case '${case_name}' failed:\n${configure_output}"
            )
        endif()
    elseif(configure_result EQUAL 0)
        message(FATAL_ERROR "Schema-tools rejection case '${case_name}' unexpectedly configured")
    elseif(NOT configure_output MATCHES "${expected_diagnostic}")
        message(FATAL_ERROR
            "Schema-tools rejection case '${case_name}' did not report "
            "'${expected_diagnostic}':\n${configure_output}"
        )
    endif()
endfunction()

run_schema_tools_case(commented_include "")
run_schema_tools_case(false_logical_path "")
run_schema_tools_case(non_embed "")
run_schema_tools_case(inferred_traits_schema "")
run_schema_tools_case(explicit_capacities "")
run_schema_tools_case(unknown_argument "received unknown arguments")
run_schema_tools_case(missing_target "requires TARGET")
run_schema_tools_case(missing_schemas "requires at least one SCHEMAS entry")
run_schema_tools_case(missing_include_directory "INCLUDE_DIRS entry does not exist")
run_schema_tools_case(unresolved_include "Could not resolve FlatBuffers include 'missing.fbs'")
run_schema_tools_case(traits_without_embed "TRAITS_HEADER requires BFBS_EMBED")
run_schema_tools_case(invalid_type "HAFS_TRAITS_ROOT_TYPE must be a qualified C\\+\\+ type")
run_schema_tools_case(invalid_capacity "HAFS_TRAITS_MAX_SERIALIZED_SIZE must be a positive integer")
run_schema_tools_case(zero_inline_capacity
    "HAFS_TRAITS_INLINE_CAPACITY must be a positive integer")
run_schema_tools_case(false_descriptor_capacity
    "HAFS_TRAITS_DESCRIPTOR_METADATA_CAPACITY must be a positive integer")
run_schema_tools_case(oversized_descriptor_capacity
    "TRAITS_DESCRIPTOR_METADATA_CAPACITY exceeds the 65536-byte descriptor")
run_schema_tools_case(ambiguous_traits_schema
    "TRAITS_SCHEMA is required when SCHEMAS contains more than one entry"
)
