# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        DIAGNOSTIC_SOURCE_DIR
        DIAGNOSTIC_BINARY_ROOT
        HOLOSCAN_FLATC_EXECUTABLE
        HOLOSCAN_FLATC_VERSION
        HOLOSCAN_SCHEMA_DIR
        HOLOSCAN_REFLECTION_SCHEMA
        HOLOSCAN_SCHEMA_TOOLS_DIR)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

foreach(diagnostic_case IN ITEMS
        missing_tensor
        object_vector
        wrong_native
        union
        recursive_table
        nested_tensor
        required_string
        required_vector
        required_table
        required_tensor
        force_align
        file_identifier)
    set(diagnostic_build_dir "${DIAGNOSTIC_BINARY_ROOT}/${diagnostic_case}")
    file(REMOVE_RECURSE "${diagnostic_build_dir}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            -S "${DIAGNOSTIC_SOURCE_DIR}"
            -B "${diagnostic_build_dir}"
            "-DDIAGNOSTIC_CASE=${diagnostic_case}"
            "-DHOLOSCAN_FLATC_EXECUTABLE=${HOLOSCAN_FLATC_EXECUTABLE}"
            "-DHOLOSCAN_FLATC_VERSION=${HOLOSCAN_FLATC_VERSION}"
            "-DHOLOSCAN_SCHEMA_DIR=${HOLOSCAN_SCHEMA_DIR}"
            "-DHOLOSCAN_REFLECTION_SCHEMA=${HOLOSCAN_REFLECTION_SCHEMA}"
            "-DHOLOSCAN_SCHEMA_TOOLS_DIR=${HOLOSCAN_SCHEMA_TOOLS_DIR}"
        RESULT_VARIABLE configure_result
        OUTPUT_VARIABLE configure_stdout
        ERROR_VARIABLE configure_stderr
    )
    set(configure_output "${configure_stdout}\n${configure_stderr}")

    if(diagnostic_case STREQUAL "wrong_native")
        if(configure_result EQUAL 0 OR
           NOT configure_output MATCHES
               "requires the generated object-API native type")
            message(FATAL_ERROR
                "Wrong-native diagnostic did not fail with the expected field-type contract:\n"
                "${configure_output}"
            )
        endif()
        continue()
    endif()
    if(NOT configure_result EQUAL 0)
        message(FATAL_ERROR
            "Could not configure ${diagnostic_case} diagnostic fixture:\n${configure_output}"
        )
    endif()

    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${diagnostic_build_dir}"
            --target diagnostic_schema_generated --parallel 4
        RESULT_VARIABLE build_result
        OUTPUT_VARIABLE build_stdout
        ERROR_VARIABLE build_stderr
    )
    set(build_output "${build_stdout}\n${build_stderr}")
    if(diagnostic_case STREQUAL "missing_tensor")
        set(expected_diagnostic "has no field of")
    elseif(diagnostic_case STREQUAL "object_vector")
        set(expected_diagnostic "vectors of tables or structs are")
    elseif(diagnostic_case STREQUAL "union")
        set(expected_diagnostic "FlatBuffers unions are unsupported")
    elseif(diagnostic_case STREQUAL "recursive_table")
        set(expected_diagnostic "recursive tables are unsupported")
    elseif(diagnostic_case STREQUAL "nested_tensor")
        set(expected_diagnostic "Tensor fields must be direct members")
    elseif(diagnostic_case MATCHES "^required_")
        set(expected_diagnostic "required FlatBuffers reference fields")
    elseif(diagnostic_case STREQUAL "force_align")
        set(expected_diagnostic "struct alignment 16 exceeds")
    elseif(diagnostic_case STREQUAL "file_identifier")
        set(expected_diagnostic "FlatBuffers file_identifier is unsupported")
    else()
        message(FATAL_ERROR "Missing expected diagnostic for ${diagnostic_case}")
    endif()
    if(build_result EQUAL 0 OR NOT build_output MATCHES "${expected_diagnostic}")
        message(FATAL_ERROR
            "${diagnostic_case} did not fail with '${expected_diagnostic}':\n${build_output}"
        )
    endif()
endforeach()
