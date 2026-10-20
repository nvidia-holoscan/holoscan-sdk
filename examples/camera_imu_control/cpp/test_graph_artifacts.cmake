# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_ARTIFACT_BUILD_DIR OR NOT DEFINED HOLOSCAN_ARTIFACT_TARGET OR
   NOT DEFINED HOLOSCAN_ARTIFACT_ROOT)
    message(FATAL_ERROR "artifact target test requires build dir, target, and root")
endif()

set(_canonical_artifact_names
    artifact-manifest.json
    endpoint-catalog.json
    execution-plan.json
    profile-manifest.json
)

function(_capture_artifact_snapshot artifact_directory output_variable)
    file(GLOB_RECURSE _artifact_entries
        LIST_DIRECTORIES TRUE
        RELATIVE "${artifact_directory}"
        "${artifact_directory}/*"
    )
    list(SORT _artifact_entries)
    set(_expected_entries "${_canonical_artifact_names}")
    list(SORT _expected_entries)
    if(NOT "${_artifact_entries}" STREQUAL "${_expected_entries}")
        message(FATAL_ERROR
            "artifact set is not canonical: expected=${_expected_entries}; "
            "actual=${_artifact_entries}")
    endif()

    set(_snapshot)
    foreach(_name IN LISTS _canonical_artifact_names)
        set(_path "${artifact_directory}/${_name}")
        file(READ "${_path}" _artifact_text)
        string(TOLOWER "${_artifact_text}" _artifact_text_lower)
        foreach(_forbidden_token IN ITEMS
                credential signing_key cuda_handle pointer file_descriptor "holomq::" iceoryx)
            string(FIND "${_artifact_text_lower}" "${_forbidden_token}" _forbidden_position)
            if(NOT _forbidden_position EQUAL -1)
                message(FATAL_ERROR
                    "${_name} exposes forbidden runtime/deployment token: ${_forbidden_token}")
            endif()
        endforeach()
        file(READ "${_path}" _bytes HEX)
        file(SHA256 "${_path}" _digest)
        file(SIZE "${_path}" _size)
        file(TIMESTAMP "${_path}" _mtime "%Y-%m-%dT%H:%M:%S" UTC)
        list(APPEND _snapshot "${_name}|${_size}|${_digest}|${_mtime}|${_bytes}")
    endforeach()
    set("${output_variable}" "${_snapshot}" PARENT_SCOPE)
endfunction()

file(REMOVE_RECURSE "${HOLOSCAN_ARTIFACT_ROOT}")
set(_artifact_build_command
    "${CMAKE_COMMAND}" --build "${HOLOSCAN_ARTIFACT_BUILD_DIR}" --parallel 4
    --target "${HOLOSCAN_ARTIFACT_TARGET}"
)
if(DEFINED HOLOSCAN_ARTIFACT_BUILD_CONFIG AND
   NOT HOLOSCAN_ARTIFACT_BUILD_CONFIG STREQUAL "")
    list(APPEND _artifact_build_command --config "${HOLOSCAN_ARTIFACT_BUILD_CONFIG}")
endif()
execute_process(
    COMMAND ${_artifact_build_command}
    RESULT_VARIABLE _first_result
    OUTPUT_VARIABLE _first_output
    ERROR_VARIABLE _first_error
    TIMEOUT 120
)
if(NOT _first_result EQUAL 0 OR
   NOT "${_first_output}${_first_error}" MATCHES "artifact-only publication complete")
    message(FATAL_ERROR
        "first artifact target execution failed (${_first_result}): "
        "${_first_output}${_first_error}")
endif()
set(_first_log "${_first_output}${_first_error}")
string(FIND "${_first_log}" " changed=1" _first_changed_position)
if(_first_changed_position EQUAL -1)
    message(FATAL_ERROR "first artifact publication did not report changed=1: ${_first_log}")
endif()
if("${_first_log}" MATCHES "camera/IMU runtime complete")
    message(FATAL_ERROR "first artifact target execution entered normal runtime mode")
endif()

file(GLOB_RECURSE _artifact_manifests
    LIST_DIRECTORIES FALSE
    "${HOLOSCAN_ARTIFACT_ROOT}/*/artifact-manifest.json"
)
list(LENGTH _artifact_manifests _artifact_manifest_count)
if(NOT _artifact_manifest_count EQUAL 1)
    message(FATAL_ERROR
        "expected exactly one artifact-manifest.json, found ${_artifact_manifest_count}")
endif()
list(GET _artifact_manifests 0 _artifact_manifest)
get_filename_component(_artifact_directory "${_artifact_manifest}" DIRECTORY)
_capture_artifact_snapshot("${_artifact_directory}" _snapshot_before)

file(READ "${_artifact_manifest}" _manifest_bytes)
string(JSON _manifest_catalog_digest
    ERROR_VARIABLE _manifest_catalog_digest_error
    GET "${_manifest_bytes}" catalog_digest)
string(LENGTH "${_manifest_catalog_digest}" _manifest_catalog_digest_length)
if(_manifest_catalog_digest_error OR
   NOT _manifest_catalog_digest_length EQUAL 71 OR
   NOT "${_manifest_catalog_digest}" MATCHES "^sha256:[0-9a-f]+$")
    message(FATAL_ERROR
        "artifact manifest has an invalid catalog_digest: "
        "${_manifest_catalog_digest_error}")
endif()
string(JSON _manifest_files_type
    ERROR_VARIABLE _manifest_files_type_error
    TYPE "${_manifest_bytes}" files)
if(_manifest_files_type_error OR NOT _manifest_files_type STREQUAL "ARRAY")
    message(FATAL_ERROR
        "artifact manifest files must be a JSON array: ${_manifest_files_type_error}")
endif()
string(JSON _manifest_file_count
    ERROR_VARIABLE _manifest_file_count_error
    LENGTH "${_manifest_bytes}" files)
if(_manifest_file_count_error OR _manifest_file_count LESS 1)
    message(FATAL_ERROR
        "artifact manifest has an invalid files array: ${_manifest_file_count_error}")
endif()
math(EXPR _manifest_file_last "${_manifest_file_count} - 1")
foreach(_name IN ITEMS endpoint-catalog.json execution-plan.json profile-manifest.json)
    file(SHA256 "${_artifact_directory}/${_name}" _digest)
    set(_manifest_match_count 0)
    foreach(_manifest_file_index RANGE 0 ${_manifest_file_last})
        string(JSON _manifest_file_name
            ERROR_VARIABLE _manifest_file_name_error
            GET "${_manifest_bytes}" files ${_manifest_file_index} name)
        string(JSON _manifest_file_digest
            ERROR_VARIABLE _manifest_file_digest_error
            GET "${_manifest_bytes}" files ${_manifest_file_index} sha256)
        if(_manifest_file_name_error OR _manifest_file_digest_error)
            message(FATAL_ERROR
                "artifact manifest files[${_manifest_file_index}] is invalid: "
                "${_manifest_file_name_error}${_manifest_file_digest_error}")
        endif()
        if("${_manifest_file_name}" STREQUAL "${_name}" AND
           "${_manifest_file_digest}" STREQUAL "sha256:${_digest}")
            math(EXPR _manifest_match_count "${_manifest_match_count} + 1")
        endif()
    endforeach()
    if(NOT _manifest_match_count EQUAL 1)
        message(FATAL_ERROR
            "artifact manifest must bind ${_name} exactly once to sha256:${_digest}; "
            "found ${_manifest_match_count} matches")
    endif()
endforeach()
string(FIND "${_first_log}" "digest=${_manifest_catalog_digest}" _catalog_digest_position)
if(_catalog_digest_position EQUAL -1)
    message(FATAL_ERROR
        "first publication did not report the canonical endpoint-catalog digest")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 2)
execute_process(
    COMMAND ${_artifact_build_command}
    RESULT_VARIABLE _second_result
    OUTPUT_VARIABLE _second_output
    ERROR_VARIABLE _second_error
    TIMEOUT 120
)
if(NOT _second_result EQUAL 0 OR
   NOT "${_second_output}${_second_error}" MATCHES "artifact-only publication complete")
    message(FATAL_ERROR
        "second artifact target execution failed (${_second_result}): "
        "${_second_output}${_second_error}")
endif()
set(_second_log "${_second_output}${_second_error}")
string(FIND "${_second_log}" " changed=0" _second_changed_position)
if(_second_changed_position EQUAL -1)
    message(FATAL_ERROR "second artifact publication did not report changed=0: ${_second_log}")
endif()
if("${_second_log}" MATCHES "camera/IMU runtime complete")
    message(FATAL_ERROR "artifact target entered normal runtime mode")
endif()
string(FIND "${_second_log}" "digest=${_manifest_catalog_digest}" _second_digest_position)
if(_second_digest_position EQUAL -1)
    message(FATAL_ERROR
        "second publication did not report the canonical endpoint-catalog digest")
endif()
_capture_artifact_snapshot("${_artifact_directory}" _snapshot_after)
if(NOT "${_snapshot_before}" STREQUAL "${_snapshot_after}")
    message(FATAL_ERROR
        "unchanged canonical artifact bytes, digests, sizes, or mtimes were not preserved")
endif()

message(STATUS
    "artifact target is cold and write-if-changed across the complete canonical file set")
