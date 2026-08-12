# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

include(CMakeParseArguments)

function(_holoscan_schema_validate_logical_path value description)
    if(value STREQUAL ""
       OR IS_ABSOLUTE "${value}"
       OR value MATCHES "(^|/)\\.\\.(/|$)"
       OR value MATCHES "(^|/)\\.(/|$)"
       OR value MATCHES "//"
       OR value MATCHES "\\\\"
       OR value MATCHES ":"
       OR value MATCHES "^/"
       OR value MATCHES "/$")
        message(FATAL_ERROR "${description} is not a canonical logical path: ${value}")
    endif()
endfunction()

function(_holoscan_schema_logical_path output file include_dirs fallback_root)
    set(logical_path)
    foreach(include_dir IN LISTS include_dirs)
        file(RELATIVE_PATH candidate "${include_dir}" "${file}")
        if(NOT IS_ABSOLUTE "${candidate}" AND NOT candidate MATCHES "^\\.\\.(/|$)")
            set(logical_path "${candidate}")
            break()
        endif()
    endforeach()
    if("${logical_path}" STREQUAL "")
        file(RELATIVE_PATH candidate "${fallback_root}" "${file}")
        if(NOT IS_ABSOLUTE "${candidate}" AND NOT candidate MATCHES "^\\.\\.(/|$)")
            set(logical_path "${candidate}")
        endif()
    endif()
    if("${logical_path}" STREQUAL "")
        message(FATAL_ERROR
            "Schema source is outside every include root and the caller source tree: ${file}"
        )
    endif()
    file(TO_CMAKE_PATH "${logical_path}" logical_path)
    _holoscan_schema_validate_logical_path("${logical_path}" "Schema source")
    set("${output}" "${logical_path}" PARENT_SCOPE)
endfunction()

# Preserve quoted schema text while replacing comments with whitespace. The include-closure scan
# remains line-oriented, but include-looking text inside either comment form cannot become a false
# dependency. Newlines are retained so diagnostics and subsequent line matching stay intuitive.
function(_holoscan_schema_strip_comments output source)
    set(result)
    set(state code)
    string(LENGTH "${source}" source_length)
    set(index 0)
    while(index LESS source_length)
        string(SUBSTRING "${source}" ${index} 1 character)
        math(EXPR next_index "${index} + 1")
        set(next_character)
        if(next_index LESS source_length)
            string(SUBSTRING "${source}" ${next_index} 1 next_character)
        endif()

        if(state STREQUAL code)
            if("${character}" STREQUAL "\"")
                string(APPEND result "${character}")
                set(state string)
            elseif("${character}" STREQUAL "/" AND "${next_character}" STREQUAL "/")
                string(APPEND result "  ")
                set(state line_comment)
                math(EXPR index "${index} + 2")
                continue()
            elseif("${character}" STREQUAL "/" AND "${next_character}" STREQUAL "*")
                string(APPEND result "  ")
                set(state block_comment)
                math(EXPR index "${index} + 2")
                continue()
            else()
                string(APPEND result "${character}")
            endif()
        elseif(state STREQUAL string)
            string(APPEND result "${character}")
            if("${character}" STREQUAL "\\" AND next_index LESS source_length)
                string(APPEND result "${next_character}")
                math(EXPR index "${index} + 2")
                continue()
            elseif("${character}" STREQUAL "\"")
                set(state code)
            endif()
        elseif(state STREQUAL line_comment)
            if("${character}" STREQUAL "\n")
                string(APPEND result "\n")
                set(state code)
            else()
                string(APPEND result " ")
            endif()
        elseif(state STREQUAL block_comment)
            if("${character}" STREQUAL "*" AND "${next_character}" STREQUAL "/")
                string(APPEND result "  ")
                set(state code)
                math(EXPR index "${index} + 2")
                continue()
            elseif("${character}" STREQUAL "\n")
                string(APPEND result "\n")
            else()
                string(APPEND result " ")
            endif()
        endif()
        math(EXPR index "${index} + 1")
    endwhile()
    set("${output}" "${result}" PARENT_SCOPE)
endfunction()

# Resolve FlatBuffers include statements at configure time. Registering the exact closure as both a
# custom-command dependency and CMAKE_CONFIGURE_DEPENDS makes an include edit regenerate evidence,
# while adding or removing an include causes CMake to reconstruct the closure before the build.
function(_holoscan_schema_collect_include_closure
         output_files output_logical_paths root_file include_dirs fallback_root)
    set(queue "${root_file}")
    set(visited)
    while(queue)
        list(POP_FRONT queue current_file)
        if(current_file IN_LIST visited)
            continue()
        endif()
        list(APPEND visited "${current_file}")
        get_filename_component(current_directory "${current_file}" DIRECTORY)
        file(READ "${current_file}" schema_source)
        _holoscan_schema_strip_comments(schema_without_comments "${schema_source}")
        string(REPLACE ";" "\\;" schema_without_comments "${schema_without_comments}")
        string(REPLACE "\n" ";" schema_lines "${schema_without_comments}")
        set(include_paths)
        foreach(schema_line IN LISTS schema_lines)
            if(schema_line MATCHES "^[ \t]*include[ \t]+\"[^\"]+\"[ \t]*;")
                string(REGEX REPLACE
                    "^[ \t]*include[ \t]+\"([^\"]+)\"[ \t]*;.*$"
                    "\\1"
                    include_path
                    "${schema_line}"
                )
                list(APPEND include_paths "${include_path}")
            endif()
        endforeach()
        foreach(include_path IN LISTS include_paths)
            set(include_candidates "${current_directory}")
            list(APPEND include_candidates ${include_dirs})
            set(resolved_include)
            foreach(include_candidate IN LISTS include_candidates)
                get_filename_component(
                    candidate_file "${include_candidate}/${include_path}" ABSOLUTE
                )
                if(EXISTS "${candidate_file}" AND NOT IS_DIRECTORY "${candidate_file}")
                    set(resolved_include "${candidate_file}")
                    break()
                endif()
            endforeach()
            if(NOT resolved_include)
                message(FATAL_ERROR
                    "Could not resolve FlatBuffers include '${include_path}' from ${current_file}"
                )
            endif()
            list(APPEND queue "${resolved_include}")
        endforeach()
    endwhile()

    list(REMOVE_ITEM visited "${root_file}")
    set(ordered_entries)
    foreach(include_file IN LISTS visited)
        _holoscan_schema_logical_path(
            include_logical_path
            "${include_file}"
            "${include_dirs}"
            "${fallback_root}"
        )
        if(include_logical_path MATCHES "\\|")
            message(FATAL_ERROR
                "Schema logical path contains reserved traits-generator delimiter: "
                "${include_logical_path}"
            )
        endif()
        list(APPEND ordered_entries "${include_logical_path}|${include_file}")
    endforeach()
    list(SORT ordered_entries)

    set(ordered_files)
    set(ordered_logical_paths)
    foreach(entry IN LISTS ordered_entries)
        string(FIND "${entry}" "|" separator)
        math(EXPR file_offset "${separator} + 1")
        string(SUBSTRING "${entry}" 0 "${separator}" logical_path)
        string(SUBSTRING "${entry}" "${file_offset}" -1 include_file)
        if(logical_path IN_LIST ordered_logical_paths)
            message(FATAL_ERROR
                "FlatBuffers include roots produce duplicate logical path '${logical_path}'"
            )
        endif()
        list(APPEND ordered_files "${include_file}")
        list(APPEND ordered_logical_paths "${logical_path}")
    endforeach()
    set("${output_files}" "${ordered_files}" PARENT_SCOPE)
    set("${output_logical_paths}" "${ordered_logical_paths}" PARENT_SCOPE)
endfunction()

# Generate FlatBuffers C++ headers through the compiler and flag contract shipped by Holoscan.
#
# TARGET names the custom target created by this function. SCHEMAS are interpreted relative to the
# caller's source directory, and OUTPUT_DIR defaults to
# <current-binary-dir>/generated/holoscan/schemas. The SDK schema directory is searched after any
# caller-provided INCLUDE_DIRS so application schemas can include SDK inputs without deriving the
# install prefix.
#
# BFBS_EMBED declares the *_bfbs_generated.h outputs. TRAITS_HEADER additionally emits a generated
# owning-value traits header for TRAITS_SCHEMA. DESCRIPTOR_TENSOR makes that header describe an
# external-buffer Tensor schema and generate its standard adapter; otherwise it describes an inline
# owning-value schema. Traits generation requires BFBS_EMBED plus explicit
# TRAITS_ROOT_TYPE, TRAITS_NATIVE_TYPE, TRAITS_BINARY_SCHEMA_TYPE, TRAITS_PACKAGE_NAMESPACE,
# TRAITS_CANONICAL_NAME, TRAITS_SCHEMA_VERSION, and TRAITS_MAX_SERIALIZED_SIZE facts. The inline and
# descriptor metadata capacities are optional positive bounds. Ordinary TRAITS_HEADER generation
# rejects any holoscan.schema.Tensor reachable from the selected root; Tensor-bearing schemas use
# the descriptor-backed schema traits surface instead. Both optional capacity defaults fall back
# to TRAITS_MAX_SERIALIZED_SIZE when omitted.
#
# Descriptor generation supports ordinary scalar, string, scalar/string vector, struct, and
# acyclic nested-table fields. Omit DESCRIPTOR_TENSOR and provide a hand-written traits header when
# a schema needs the standard factory with a custom codec for another object-API shape.
#
# The function publishes <target>_GENERATED_HEADERS and <target>_TRAITS_HEADER in the caller's
# scope. The generated target records the configured package version in the
# HOLOSCAN_FLATC_VERSION_CONFIGURED property.
function(holoscan_add_flatbuffer_schema)
    set(options BFBS_EMBED DESCRIPTOR_TENSOR)
    set(one_value_args
        TARGET
        OUTPUT_DIR
        TRAITS_HEADER
        TRAITS_SCHEMA
        TRAITS_ROOT_TYPE
        TRAITS_NATIVE_TYPE
        TRAITS_BINARY_SCHEMA_TYPE
        TRAITS_PACKAGE_NAMESPACE
        TRAITS_CANONICAL_NAME
        TRAITS_SCHEMA_VERSION
        TRAITS_MAX_SERIALIZED_SIZE
        TRAITS_INLINE_CAPACITY
        TRAITS_DESCRIPTOR_METADATA_CAPACITY
    )
    set(multi_value_args SCHEMAS INCLUDE_DIRS)
    cmake_parse_arguments(HAFS
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )

    if(HAFS_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema received unknown arguments: "
            "${HAFS_UNPARSED_ARGUMENTS}"
        )
    endif()
    if(NOT HAFS_TARGET)
        message(FATAL_ERROR "holoscan_add_flatbuffer_schema requires TARGET")
    endif()
    if(TARGET "${HAFS_TARGET}")
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema TARGET '${HAFS_TARGET}' already exists"
        )
    endif()
    if(NOT HAFS_SCHEMAS)
        message(FATAL_ERROR "holoscan_add_flatbuffer_schema requires at least one SCHEMAS entry")
    endif()
    set(_holoscan_schema_traits_arguments
        HAFS_TRAITS_SCHEMA
        HAFS_TRAITS_ROOT_TYPE
        HAFS_TRAITS_NATIVE_TYPE
        HAFS_TRAITS_BINARY_SCHEMA_TYPE
        HAFS_TRAITS_PACKAGE_NAMESPACE
        HAFS_TRAITS_CANONICAL_NAME
        HAFS_TRAITS_SCHEMA_VERSION
        HAFS_TRAITS_MAX_SERIALIZED_SIZE
        HAFS_TRAITS_INLINE_CAPACITY
        HAFS_TRAITS_DESCRIPTOR_METADATA_CAPACITY
    )
    if(NOT HAFS_TRAITS_HEADER)
        if(HAFS_DESCRIPTOR_TENSOR)
            message(FATAL_ERROR
                "holoscan_add_flatbuffer_schema DESCRIPTOR_TENSOR requires TRAITS_HEADER"
            )
        endif()
        foreach(_holoscan_schema_traits_argument IN LISTS _holoscan_schema_traits_arguments)
            if(DEFINED "${_holoscan_schema_traits_argument}"
               AND NOT "${${_holoscan_schema_traits_argument}}" STREQUAL "")
                message(FATAL_ERROR
                    "${_holoscan_schema_traits_argument} requires TRAITS_HEADER"
                )
            endif()
        endforeach()
    endif()
    if(NOT TARGET flatbuffers::flatc)
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema requires flatbuffers::flatc"
        )
    endif()
    if(NOT TARGET holoscan::fbs_tools)
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema requires holoscan::fbs_tools"
        )
    endif()
    set(_holoscan_schema_flatc_target flatbuffers::flatc)
    get_target_property(
        _holoscan_schema_flatc_version
        holoscan::fbs_tools
        HOLOSCAN_FLATC_VERSION
    )
    get_target_property(
        _holoscan_sdk_schema_dir
        holoscan::fbs_tools
        HOLOSCAN_SCHEMA_DIR
    )
    get_target_property(
        _holoscan_flatbuffers_reflection_schema
        holoscan::fbs_tools
        HOLOSCAN_FLATBUFFERS_REFLECTION_SCHEMA
    )
    foreach(_holoscan_schema_required_property IN ITEMS
            _holoscan_schema_flatc_version
            _holoscan_sdk_schema_dir
            _holoscan_flatbuffers_reflection_schema)
        if("${${_holoscan_schema_required_property}}" STREQUAL "" OR
           "${${_holoscan_schema_required_property}}" MATCHES "-NOTFOUND$")
            message(FATAL_ERROR
                "holoscan::fbs_tools does not provide required Holoscan schema metadata"
            )
        endif()
    endforeach()

    if(NOT HAFS_OUTPUT_DIR)
        set(HAFS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/holoscan/schemas")
    endif()
    get_filename_component(
        _holoscan_schema_output_dir "${HAFS_OUTPUT_DIR}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
    )

    list(APPEND HAFS_INCLUDE_DIRS "${_holoscan_sdk_schema_dir}")
    list(REMOVE_DUPLICATES HAFS_INCLUDE_DIRS)

    set(_holoscan_schema_include_args)
    set(_holoscan_schema_include_dirs_abs)
    foreach(_holoscan_schema_include_dir IN LISTS HAFS_INCLUDE_DIRS)
        get_filename_component(
            _holoscan_schema_include_abs "${_holoscan_schema_include_dir}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        if(NOT IS_DIRECTORY "${_holoscan_schema_include_abs}")
            message(FATAL_ERROR
                "holoscan_add_flatbuffer_schema INCLUDE_DIRS entry does not exist: "
                "${_holoscan_schema_include_abs}"
            )
        endif()
        list(APPEND _holoscan_schema_include_args -I "${_holoscan_schema_include_abs}")
        list(APPEND _holoscan_schema_include_dirs_abs "${_holoscan_schema_include_abs}")
    endforeach()

    set(_holoscan_schema_inputs)
    set(_holoscan_schema_dependencies)
    set(_holoscan_schema_generated_headers)
    set(_holoscan_schema_output_stems)
    set(_holoscan_schema_inputs_abs)
    set(_holoscan_schema_inputs_logical)
    foreach(_holoscan_schema IN LISTS HAFS_SCHEMAS)
        get_filename_component(
            _holoscan_schema_abs "${_holoscan_schema}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        if(NOT EXISTS "${_holoscan_schema_abs}" OR IS_DIRECTORY "${_holoscan_schema_abs}")
            message(FATAL_ERROR
                "holoscan_add_flatbuffer_schema SCHEMAS entry is not a file: "
                "${_holoscan_schema_abs}"
            )
        endif()
        get_filename_component(_holoscan_schema_extension "${_holoscan_schema_abs}" EXT)
        if(NOT _holoscan_schema_extension STREQUAL ".fbs")
            message(FATAL_ERROR
                "holoscan_add_flatbuffer_schema SCHEMAS entry must end in .fbs: "
                "${_holoscan_schema_abs}"
            )
        endif()
        get_filename_component(_holoscan_schema_stem "${_holoscan_schema_abs}" NAME_WE)
        if(_holoscan_schema_stem IN_LIST _holoscan_schema_output_stems)
            message(FATAL_ERROR
                "holoscan_add_flatbuffer_schema SCHEMAS entries have duplicate output stem "
                "'${_holoscan_schema_stem}'"
            )
        endif()
        list(APPEND _holoscan_schema_output_stems "${_holoscan_schema_stem}")
        list(APPEND _holoscan_schema_inputs "${_holoscan_schema}")
        list(APPEND _holoscan_schema_inputs_abs "${_holoscan_schema_abs}")
        list(APPEND _holoscan_schema_dependencies "${_holoscan_schema_abs}")
        _holoscan_schema_logical_path(
            _holoscan_schema_logical_path
            "${_holoscan_schema_abs}"
            "${_holoscan_schema_include_dirs_abs}"
            "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        list(APPEND _holoscan_schema_inputs_logical "${_holoscan_schema_logical_path}")
        _holoscan_schema_collect_include_closure(
            _holoscan_schema_closure_files
            _holoscan_schema_closure_logical_paths
            "${_holoscan_schema_abs}"
            "${_holoscan_schema_include_dirs_abs}"
            "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        list(APPEND _holoscan_schema_dependencies ${_holoscan_schema_closure_files})
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${_holoscan_schema_abs}" ${_holoscan_schema_closure_files}
        )
        if(HAFS_BFBS_EMBED)
            list(APPEND _holoscan_schema_generated_headers
                "${_holoscan_schema_output_dir}/${_holoscan_schema_stem}_bfbs_generated.h"
            )
        endif()
        list(APPEND _holoscan_schema_generated_headers
            "${_holoscan_schema_output_dir}/${_holoscan_schema_stem}_generated.h"
        )
    endforeach()
    list(REMOVE_DUPLICATES _holoscan_schema_dependencies)

    # Keep both the flatc invocation and published package evidence in one stable order:
    # SDK-pinned code-generation flags, the normalized output placeholder, include roots in
    # caller order, then root schemas in declaration order. Applications select inputs and roots
    # but do not reorder or override the SDK-pinned object-API, reflection, or embedding flags.
    set(_holoscan_schema_flatc_flags)
    if(HAFS_BFBS_EMBED)
        list(APPEND _holoscan_schema_flatc_flags --bfbs-gen-embed)
    endif()
    if(HAFS_DESCRIPTOR_TENSOR)
        # Canonical Tensor ordering is the FlatBuffers field-id order. Requiring explicit IDs
        # prevents source declaration order from becoming an accidental public convention.
        list(APPEND _holoscan_schema_flatc_flags --require-explicit-ids)
    endif()
    list(APPEND _holoscan_schema_flatc_flags
        --cpp
        --cpp-ptr-type std::shared_ptr
        --gen-object-api
        --reflect-names
        --reflect-types
    )
    if(HAFS_BFBS_EMBED)
        list(APPEND _holoscan_schema_flatc_flags --schema)
    endif()

    set(_holoscan_schema_traits_header)
    set(_holoscan_schema_traits_commands)
    set(_holoscan_schema_traits_config)
    if(HAFS_TRAITS_HEADER)
        if(NOT HAFS_BFBS_EMBED)
            message(FATAL_ERROR
                "holoscan_add_flatbuffer_schema TRAITS_HEADER requires BFBS_EMBED"
            )
        endif()
        foreach(_holoscan_schema_required_traits_argument IN ITEMS
                TRAITS_ROOT_TYPE
                TRAITS_NATIVE_TYPE
                TRAITS_BINARY_SCHEMA_TYPE
                TRAITS_PACKAGE_NAMESPACE
                TRAITS_CANONICAL_NAME
                TRAITS_SCHEMA_VERSION
                TRAITS_MAX_SERIALIZED_SIZE)
            if(NOT HAFS_${_holoscan_schema_required_traits_argument})
                message(FATAL_ERROR
                    "holoscan_add_flatbuffer_schema TRAITS_HEADER requires "
                    "${_holoscan_schema_required_traits_argument}"
                )
            endif()
        endforeach()
        if(IS_ABSOLUTE "${HAFS_TRAITS_HEADER}"
           OR HAFS_TRAITS_HEADER MATCHES "(^|/)\\.\\.(/|$)"
           OR HAFS_TRAITS_HEADER MATCHES "\\\\"
           OR NOT HAFS_TRAITS_HEADER MATCHES "\\.(h|hh|hpp|hxx)$")
            message(FATAL_ERROR
                "TRAITS_HEADER must be a relative header path below OUTPUT_DIR: "
                "${HAFS_TRAITS_HEADER}"
            )
        endif()
        foreach(_holoscan_schema_type IN ITEMS
                HAFS_TRAITS_ROOT_TYPE
                HAFS_TRAITS_NATIVE_TYPE
                HAFS_TRAITS_BINARY_SCHEMA_TYPE)
            if(NOT "${${_holoscan_schema_type}}" MATCHES
                   "^(::)?[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z_][A-Za-z0-9_]*)+$")
                message(FATAL_ERROR
                    "${_holoscan_schema_type} must be a qualified C++ type: "
                    "${${_holoscan_schema_type}}"
                )
            endif()
        endforeach()
        if(HAFS_TRAITS_ROOT_TYPE STREQUAL HAFS_TRAITS_NATIVE_TYPE)
            message(FATAL_ERROR "TRAITS_ROOT_TYPE and TRAITS_NATIVE_TYPE must be distinct")
        endif()
        if(HAFS_DESCRIPTOR_TENSOR)
            if(NOT HAFS_TRAITS_NATIVE_TYPE STREQUAL "${HAFS_TRAITS_ROOT_TYPE}T")
                message(FATAL_ERROR
                    "DESCRIPTOR_TENSOR requires the generated object-API native type "
                    "TRAITS_NATIVE_TYPE=${HAFS_TRAITS_ROOT_TYPE}T"
                )
            endif()
            if(NOT EXISTS "${_holoscan_flatbuffers_reflection_schema}")
                message(FATAL_ERROR
                    "DESCRIPTOR_TENSOR requires the FlatBuffers reflection schema from "
                    "the matching SDK FlatBuffers package"
                )
            endif()
        endif()
        if(NOT HAFS_TRAITS_PACKAGE_NAMESPACE MATCHES
               "^[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z_][A-Za-z0-9_]*)*$")
            message(FATAL_ERROR
                "TRAITS_PACKAGE_NAMESPACE must be a C++ namespace: "
                "${HAFS_TRAITS_PACKAGE_NAMESPACE}"
            )
        endif()
        if(HAFS_TRAITS_CANONICAL_NAME MATCHES "[\"\\\\\r\n]"
           OR NOT HAFS_TRAITS_CANONICAL_NAME MATCHES "\\.")
            message(FATAL_ERROR
                "TRAITS_CANONICAL_NAME must be a globally scoped C++-style schema name"
            )
        endif()
        foreach(_holoscan_schema_positive_integer IN ITEMS
                HAFS_TRAITS_SCHEMA_VERSION
                HAFS_TRAITS_MAX_SERIALIZED_SIZE)
            if(NOT "${${_holoscan_schema_positive_integer}}" MATCHES "^[1-9][0-9]*$")
                message(FATAL_ERROR
                    "${_holoscan_schema_positive_integer} must be a positive integer"
                )
            endif()
        endforeach()
        if(HAFS_DESCRIPTOR_TENSOR AND
           HAFS_TRAITS_DESCRIPTOR_METADATA_CAPACITY GREATER 65536)
            message(FATAL_ERROR
                "TRAITS_DESCRIPTOR_METADATA_CAPACITY exceeds the 65536-byte descriptor metadata "
                "generation bound"
            )
        endif()
        if("${HAFS_TRAITS_INLINE_CAPACITY}" STREQUAL "")
            set(HAFS_TRAITS_INLINE_CAPACITY "${HAFS_TRAITS_MAX_SERIALIZED_SIZE}")
        endif()
        if("${HAFS_TRAITS_DESCRIPTOR_METADATA_CAPACITY}" STREQUAL "")
            set(HAFS_TRAITS_DESCRIPTOR_METADATA_CAPACITY
                "${HAFS_TRAITS_MAX_SERIALIZED_SIZE}"
            )
        endif()
        foreach(_holoscan_schema_positive_integer IN ITEMS
                HAFS_TRAITS_INLINE_CAPACITY
                HAFS_TRAITS_DESCRIPTOR_METADATA_CAPACITY)
            if(NOT "${${_holoscan_schema_positive_integer}}" MATCHES "^[1-9][0-9]*$")
                message(FATAL_ERROR
                    "${_holoscan_schema_positive_integer} must be a positive integer"
                )
            endif()
        endforeach()

        if(NOT HAFS_TRAITS_SCHEMA)
            list(LENGTH HAFS_SCHEMAS _holoscan_schema_count)
            if(NOT _holoscan_schema_count EQUAL 1)
                message(FATAL_ERROR
                    "TRAITS_SCHEMA is required when SCHEMAS contains more than one entry"
                )
            endif()
            list(GET HAFS_SCHEMAS 0 HAFS_TRAITS_SCHEMA)
        endif()
        get_filename_component(
            _holoscan_schema_traits_input_abs "${HAFS_TRAITS_SCHEMA}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        list(FIND _holoscan_schema_inputs_abs
            "${_holoscan_schema_traits_input_abs}" _holoscan_schema_traits_index
        )
        if(_holoscan_schema_traits_index EQUAL -1)
            message(FATAL_ERROR "TRAITS_SCHEMA must name one SCHEMAS entry")
        endif()
        if(IS_ABSOLUTE "${HAFS_TRAITS_SCHEMA}")
            message(FATAL_ERROR
                "TRAITS_SCHEMA must use a canonical relative spelling so .bfbs filenames "
                "do not retain a host path"
            )
        endif()
        _holoscan_schema_validate_logical_path(
            "${HAFS_TRAITS_SCHEMA}" "TRAITS_SCHEMA"
        )
        list(GET _holoscan_schema_output_stems
            "${_holoscan_schema_traits_index}" _holoscan_schema_traits_stem
        )
        list(GET _holoscan_schema_inputs_logical
            "${_holoscan_schema_traits_index}" _holoscan_schema_traits_root_logical
        )
        _holoscan_schema_collect_include_closure(
            _holoscan_schema_traits_include_files
            _holoscan_schema_traits_include_logical_paths
            "${_holoscan_schema_traits_input_abs}"
            "${_holoscan_schema_include_dirs_abs}"
            "${CMAKE_CURRENT_SOURCE_DIR}"
        )

        set(_holoscan_schema_traits_ordered_flags ${_holoscan_schema_flatc_flags})
        list(APPEND _holoscan_schema_traits_ordered_flags
            -o "<generated-schema-dir>"
        )
        set(_holoscan_schema_traits_include_root_mapping)
        set(_holoscan_schema_include_root_index 0)
        foreach(_holoscan_schema_include_dir IN LISTS _holoscan_schema_include_dirs_abs)
            list(APPEND _holoscan_schema_traits_ordered_flags
                -I "<schema-root-${_holoscan_schema_include_root_index}>"
            )
            if(_holoscan_schema_traits_include_root_mapping)
                string(APPEND _holoscan_schema_traits_include_root_mapping ";")
            endif()
            string(APPEND _holoscan_schema_traits_include_root_mapping
                "schema-root-${_holoscan_schema_include_root_index}=."
            )
            math(EXPR _holoscan_schema_include_root_index
                "${_holoscan_schema_include_root_index} + 1"
            )
        endforeach()
        list(APPEND _holoscan_schema_traits_ordered_flags
            ${_holoscan_schema_inputs_logical}
        )

        set(_holoscan_schema_traits_header
            "${_holoscan_schema_output_dir}/${HAFS_TRAITS_HEADER}"
        )
        set(_holoscan_schema_traits_metadata_dir
            "${CMAKE_CURRENT_BINARY_DIR}/${HAFS_TARGET}_schema_traits"
        )
        set(_holoscan_schema_traits_bfbs
            "${_holoscan_schema_traits_metadata_dir}/${_holoscan_schema_traits_stem}.bfbs"
        )
        set(_holoscan_schema_traits_config
            "${_holoscan_schema_traits_metadata_dir}/traits-config.cmake"
        )

        set(HST_OUTPUT_FILE "${_holoscan_schema_traits_header}")
        set(HST_BFBS_FILE "${_holoscan_schema_traits_bfbs}")
        set(HST_EXPECTED_FLATC_VERSION "${_holoscan_schema_flatc_version}")
        set(HST_FLATBUFFERS_REFLECTION_SCHEMA
            "${_holoscan_flatbuffers_reflection_schema}"
        )
        set(HST_ROOT_GENERATED_HEADER
            "${_holoscan_schema_traits_stem}_generated.h"
        )
        set(HST_BFBS_GENERATED_HEADER
            "${_holoscan_schema_traits_stem}_bfbs_generated.h"
        )
        set(HST_BFBS_GENERATED_FILE
            "${_holoscan_schema_output_dir}/${_holoscan_schema_traits_stem}_bfbs_generated.h"
        )
        set(HST_ROOT_TYPE "${HAFS_TRAITS_ROOT_TYPE}")
        set(HST_NATIVE_TYPE "${HAFS_TRAITS_NATIVE_TYPE}")
        set(HST_BINARY_SCHEMA_TYPE "${HAFS_TRAITS_BINARY_SCHEMA_TYPE}")
        set(HST_PACKAGE_NAMESPACE "${HAFS_TRAITS_PACKAGE_NAMESPACE}")
        set(HST_CANONICAL_NAME "${HAFS_TRAITS_CANONICAL_NAME}")
        set(HST_SCHEMA_VERSION "${HAFS_TRAITS_SCHEMA_VERSION}")
        set(HST_MAX_SERIALIZED_SIZE "${HAFS_TRAITS_MAX_SERIALIZED_SIZE}")
        set(HST_INLINE_CAPACITY "${HAFS_TRAITS_INLINE_CAPACITY}")
        set(HST_DESCRIPTOR_METADATA_CAPACITY
            "${HAFS_TRAITS_DESCRIPTOR_METADATA_CAPACITY}"
        )
        set(HST_ROOT_LOGICAL_PATH "${_holoscan_schema_traits_root_logical}")
        set(HST_INCLUDE_FILES "${_holoscan_schema_traits_include_files}")
        set(HST_INCLUDE_LOGICAL_PATHS
            "${_holoscan_schema_traits_include_logical_paths}"
        )
        set(HST_ORDERED_FLAGS "${_holoscan_schema_traits_ordered_flags}")
        set(HST_INCLUDE_ROOT_MAPPING
            "${_holoscan_schema_traits_include_root_mapping}"
        )
        set(HST_HOST_PATHS
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${_holoscan_schema_output_dir}"
            "${_holoscan_schema_include_dirs_abs}"
            "${_holoscan_schema_inputs_abs}"
        )
        set(HST_DESCRIPTOR_TENSOR "${HAFS_DESCRIPTOR_TENSOR}")
        set(HST_FLATBUFFERS_REFLECTION_SCHEMA
            "${_holoscan_flatbuffers_reflection_schema}"
        )
        file(MAKE_DIRECTORY "${_holoscan_schema_traits_metadata_dir}")
        configure_file(
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/HoloscanSchemaTraitsConfig.cmake.in"
            "${_holoscan_schema_traits_config}"
            @ONLY
        )
        list(APPEND _holoscan_schema_generated_headers
            "${_holoscan_schema_traits_header}"
        )
        set(_holoscan_schema_traits_dependencies)
        if(HAFS_DESCRIPTOR_TENSOR)
            list(APPEND _holoscan_schema_traits_dependencies
                "${_holoscan_flatbuffers_reflection_schema}"
            )
        endif()
        set(_holoscan_schema_traits_commands
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${_holoscan_schema_traits_metadata_dir}"
            COMMAND "${_holoscan_schema_flatc_target}"
                --binary
                --schema
                -o "${_holoscan_schema_traits_metadata_dir}"
                ${_holoscan_schema_include_args}
                "${HAFS_TRAITS_SCHEMA}"
            COMMAND "${CMAKE_COMMAND}"
                "-DHST_CONFIG_FILE=${_holoscan_schema_traits_config}"
                "-DHST_FLATC_EXECUTABLE=$<TARGET_FILE:${_holoscan_schema_flatc_target}>"
                -P
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/HoloscanGenerateSchemaTraits.cmake"
        )
        set(_holoscan_schema_traits_byproducts "${_holoscan_schema_traits_bfbs}")
    endif()

    add_custom_command(
        OUTPUT ${_holoscan_schema_generated_headers}
        BYPRODUCTS ${_holoscan_schema_traits_byproducts}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_holoscan_schema_output_dir}"
        COMMAND "${_holoscan_schema_flatc_target}"
            ${_holoscan_schema_flatc_flags}
            -o "${_holoscan_schema_output_dir}"
            ${_holoscan_schema_include_args}
            ${_holoscan_schema_inputs}
        ${_holoscan_schema_traits_commands}
        DEPENDS
            ${_holoscan_schema_flatc_target}
            ${_holoscan_schema_dependencies}
            ${_holoscan_schema_traits_dependencies}
            ${_holoscan_schema_traits_config}
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/HoloscanGenerateSchemaTraits.cmake"
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/HoloscanGenerateDescriptorSchemaTraits.cmake"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT
            "Generating ${HAFS_TARGET} with flatc ${_holoscan_schema_flatc_version}"
        VERBATIM
    )
    add_custom_target("${HAFS_TARGET}" DEPENDS ${_holoscan_schema_generated_headers})
    set_target_properties("${HAFS_TARGET}" PROPERTIES
        HOLOSCAN_FLATBUFFER_GENERATED_HEADERS "${_holoscan_schema_generated_headers}"
        HOLOSCAN_FLATBUFFER_OUTPUT_DIR "${_holoscan_schema_output_dir}"
        HOLOSCAN_FLATC_VERSION_CONFIGURED "${_holoscan_schema_flatc_version}"
        HOLOSCAN_FLATBUFFER_TRAITS_HEADER "${_holoscan_schema_traits_header}"
    )

    set("${HAFS_TARGET}_GENERATED_HEADERS"
        "${_holoscan_schema_generated_headers}"
        PARENT_SCOPE
    )
    set("${HAFS_TARGET}_TRAITS_HEADER"
        "${_holoscan_schema_traits_header}"
        PARENT_SCOPE
    )
endfunction()
