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

function(run_schema_tools_build_case case_name expected_diagnostic)
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
    if(NOT configure_result EQUAL 0)
        message(FATAL_ERROR
            "Schema-tools build case '${case_name}' did not configure:\n"
            "${configure_stdout}${configure_stderr}"
        )
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${case_build_dir}"
            --target "schema_tools_${case_name}"
        RESULT_VARIABLE build_result
        OUTPUT_VARIABLE build_stdout
        ERROR_VARIABLE build_stderr
    )
    set(build_output "${build_stdout}${build_stderr}")
    if(expected_diagnostic STREQUAL "")
        if(NOT build_result EQUAL 0)
            message(FATAL_ERROR
                "Schema-tools build case '${case_name}' failed:\n${build_output}"
            )
        endif()
    elseif(build_result EQUAL 0)
        message(FATAL_ERROR "Schema-tools build case '${case_name}' unexpectedly succeeded")
    elseif(NOT build_output MATCHES "${expected_diagnostic}")
        message(FATAL_ERROR
            "Schema-tools build case '${case_name}' did not report "
            "'${expected_diagnostic}':\n${build_output}"
        )
    endif()
endfunction()

run_schema_tools_build_case(commented_include "")
run_schema_tools_build_case(false_logical_path "")
run_schema_tools_build_case(non_embed "")
run_schema_tools_build_case(embed "")
set(embed_header
    "${HOLOSCAN_SCHEMA_TEST_WORK_DIR}/embed/generated/holoscan/schemas/valid_bfbs_generated.h"
)
if(NOT EXISTS "${embed_header}")
    message(FATAL_ERROR "BFBS_EMBED did not generate ${embed_header}")
endif()
set(embed_traits_header
    "${HOLOSCAN_SCHEMA_TEST_WORK_DIR}/embed/generated/holoscan/schemas/valid_schema_traits.hpp"
)
file(READ "${embed_traits_header}" embed_traits_contents)
if(NOT embed_traits_contents MATCHES
   "using binary_schema_type = schema_tools::RootBinarySchema;")
    message(FATAL_ERROR
        "BFBS_EMBED did not bind reflection bytes to the generated root in ${embed_traits_header}"
    )
endif()
run_schema_tools_build_case(inferred_traits_schema "")
run_schema_tools_build_case(conformance "")
run_schema_tools_build_case(provenance "")
set(provenance_file "${HOLOSCAN_SCHEMA_TEST_WORK_DIR}/provenance/schema.provenance.json")
if(NOT EXISTS "${provenance_file}")
    message(FATAL_ERROR "Schema provenance sidecar was not generated")
endif()
file(READ "${provenance_file}" provenance_json)
foreach(required_member IN ITEMS format flatc_build flatc_flags bfbs_sha256 root includes)
    string(JSON member ERROR_VARIABLE json_error GET "${provenance_json}" "${required_member}")
    if(json_error)
        message(FATAL_ERROR "Schema provenance sidecar lacks ${required_member}: ${json_error}")
    endif()
endforeach()
run_schema_tools_build_case(multi_root_conformance "")
run_schema_tools_build_case(implicit_ids "all fields must have an 'id' attribute")
run_schema_tools_build_case(incompatible_conformance "schemas don't conform")
run_schema_tools_case(ambiguous_conformance_schema
    "CONFORMANCE_SCHEMA when SCHEMAS has multiple"
)
run_schema_tools_case(nonmember_conformance_schema
    "CONFORMANCE_SCHEMA must name one SCHEMAS entry"
)
run_schema_tools_case(unknown_argument "received unknown arguments")
run_schema_tools_case(missing_target "requires TARGET")
run_schema_tools_case(missing_schemas "requires SCHEMA or SCHEMAS")
run_schema_tools_case(missing_include_directory "INCLUDE_DIRS entry does not exist")
run_schema_tools_build_case(unresolved_include "unable to load include file")
run_schema_tools_build_case(traits_without_embed "")
# A schema that did not opt in must not name reflection bytes, so holoscan::binary_schema<T>()
# stays unavailable for it rather than resolving to another schema's bytes.
set(non_embed_traits_header
    "${HOLOSCAN_SCHEMA_TEST_WORK_DIR}/traits_without_embed/generated/holoscan/schemas/valid_schema_traits.hpp"
)
file(READ "${non_embed_traits_header}" non_embed_traits_contents)
if(non_embed_traits_contents MATCHES "binary_schema_type")
    message(FATAL_ERROR
        "Traits generated without BFBS_EMBED named reflection bytes in ${non_embed_traits_header}"
    )
endif()
run_schema_tools_case(invalid_capacity "MAX_SERIALIZED_SIZE must be a positive integer")
run_schema_tools_case(ambiguous_traits_schema
    "ROOT_SCHEMA is required when SCHEMAS contains more than one entry"
)
