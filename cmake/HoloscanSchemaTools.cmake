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

# Generate FlatBuffers C++ headers through the compiler and flag contract shipped by Holoscan.
#
# TARGET names the custom target created by this function. SCHEMAS are interpreted relative to the
# caller's source directory, and OUTPUT_DIR defaults to
# <current-binary-dir>/generated/holoscan/schemas. The SDK schema directory is searched after any
# caller-provided INCLUDE_DIRS so application schemas can include SDK inputs without deriving the
# install prefix.
#
# BFBS_EMBED opts into runtime *_bfbs_generated.h outputs. Combined with traits generation it also
# binds those reflection bytes to the generated root, so holoscan::binary_schema<T>() returns them
# for that schema and no other. The bytes stay out of schema identity, hashing, and compatibility.
# PYTHON additionally emits a generated dataclass module, typing stub, and compiled
# payload-registration adapter. PYTHON_OUTPUT_DIR can place those package artifacts separately from
# the C++ outputs. PYTHON_VALUES_ONLY emits only a dataclass module and stub for a rootless helper
# schema. PYTHON_EXTERNAL_TYPES entries use Schema.Type=python.module to reuse canonical values.
# PYTHON_CONSTANTS entries use PUBLIC_NAME=INTEGER to publish intentional module-level constants in
# both the generated runtime module and stub. PYTHON_LINK_LIBRARIES supplies a component-owned custom
# descriptor codec to the registration extension. PYTHON_TRAITS_INCLUDE replaces the generated
# traits include in that extension when a component-owned public header also declares the custom
# runtime operations. PYTHON_CODEC_DEFERRED lets the owning component schedule that extension after
# its native codec target exists. Supplying PYTHON_INSTALL_COMPONENT registers a production schema
# once for build, install, and wheel consumers. MAX_SERIALIZED_SIZE emits a generated owning-value
# traits header
# for SCHEMA. A direct holoscan.schema.Tensor field makes that header describe an external-buffer
# Tensor schema and generate its standard adapter; otherwise it describes an inline owning-value
# schema. DESCRIPTOR_TENSOR_CUSTOM_CODEC retains the generated
# descriptor identity, field inventory, package, and native traits while leaving Runtime type
# operations to a separately authored codec. The selected root, owning object-API type, canonical
# name, file identifier, and private generated package namespace are inferred from the build-tree
# BFBS. Traits generation requires only an explicit MAX_SERIALIZED_SIZE policy bound.
# COMPATIBILITY_EPOCH is an optional positive intentional wire-break epoch and defaults to one.
# MAX_SERIALIZED_SIZE bounds the complete payload for an ordinary schema and the metadata-only
# FlatBuffer for a descriptor-Tensor schema. Tensor must be a direct field of the selected root;
# nested Tensor fields are rejected. CONFORMANCE_BASELINE optionally names a
# committed source .fbs checked with the pinned flatc before generation.
# CONFORMANCE_SCHEMA selects the checked root and is required for a multi-root invocation.
# PROVENANCE_FILE optionally writes exact build evidence to a JSON sidecar. It never enters the
# generated C++ traits, schema identity, Graph, execution plan, catalog, or wire format.
# NATIVE_BOUNDS_FILE opts into generated reusable native storage. This source-relative JSON file
# authors per-field string/vector limits and a charged native-memory budget, independently of
# MAX_SERIALIZED_SIZE. It does not change native field types or wire identity. Custom codecs
# cannot opt into generated preparation. Unspecified schemas retain their existing owning path.
#
# Descriptor generation supports ordinary scalar, string, scalar/string vector, struct, and
# acyclic nested-table fields. DESCRIPTOR_TENSOR_CUSTOM_CODEC keeps generated identity and field
# inventory while a component supplies the standard factory with its own ordinary-field codec.
#
# SCHEMA is the normal one-root form. SCHEMAS, ROOT_SCHEMA, OUTPUT_DIR, TRAITS_HEADER, INCLUDE_DIRS,
# and the conformance/provenance options are advanced controls. The function creates an interface
# library that propagates generation ordering, its generated include directory, and required
# FlatBuffers/Holoscan schema dependencies. Generated outputs and the configured package version
# are published as HOLOSCAN_FLATBUFFER_* target properties.
function(holoscan_add_flatbuffer_schema)
    set(options
        BFBS_EMBED
        DESCRIPTOR_TENSOR_CUSTOM_CODEC
        PYTHON
        PYTHON_CODEC_DEFERRED
        PYTHON_VALUES_ONLY)
    set(one_value_args
        TARGET
        SCHEMA
        OUTPUT_DIR
        PYTHON_INSTALL_COMPONENT
        PYTHON_OUTPUT_DIR
        PYTHON_TRAITS_INCLUDE
        TRAITS_HEADER
        ROOT_SCHEMA
        COMPATIBILITY_EPOCH
        MAX_SERIALIZED_SIZE
        CONFORMANCE_SCHEMA
        CONFORMANCE_BASELINE
        PROVENANCE_FILE
        NATIVE_BOUNDS_FILE
    )
    set(multi_value_args
        SCHEMAS
        INCLUDE_DIRS
        PYTHON_CONSTANTS
        PYTHON_EXTERNAL_TYPES
        PYTHON_LINK_LIBRARIES
    )
    cmake_parse_arguments(HAFS
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )

    if(HAFS_PYTHON)
        find_package(Python3 3.12 REQUIRED
            COMPONENTS Interpreter Development.Module Development.SABIModule)
    else()
        # Native-only generation uses the same interpreter floor as the rest of the SDK.
        find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
    endif()

    set(_holoscan_schema_traits_requested FALSE)
    if(DEFINED HAFS_MAX_SERIALIZED_SIZE AND NOT "${HAFS_MAX_SERIALIZED_SIZE}" STREQUAL "")
        set(_holoscan_schema_traits_requested TRUE)
    endif()

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
    if(HAFS_SCHEMA AND HAFS_SCHEMAS)
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema accepts SCHEMA or SCHEMAS, not both"
        )
    endif()
    if(HAFS_SCHEMA)
        set(HAFS_SCHEMAS "${HAFS_SCHEMA}")
    endif()
    if(NOT HAFS_SCHEMAS)
        message(FATAL_ERROR "holoscan_add_flatbuffer_schema requires SCHEMA or SCHEMAS")
    endif()
    if(HAFS_DESCRIPTOR_TENSOR_CUSTOM_CODEC AND NOT _holoscan_schema_traits_requested)
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema DESCRIPTOR_TENSOR_CUSTOM_CODEC requires "
            "MAX_SERIALIZED_SIZE"
        )
    endif()
    if(HAFS_PYTHON AND NOT _holoscan_schema_traits_requested)
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema PYTHON requires MAX_SERIALIZED_SIZE"
        )
    endif()
    if(HAFS_PYTHON_VALUES_ONLY AND HAFS_PYTHON)
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema accepts PYTHON or PYTHON_VALUES_ONLY, not both"
        )
    endif()
    if(HAFS_PYTHON_CODEC_DEFERRED AND NOT HAFS_PYTHON)
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema PYTHON_CODEC_DEFERRED requires PYTHON"
        )
    endif()
    if(HAFS_PYTHON_VALUES_ONLY AND _holoscan_schema_traits_requested)
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema PYTHON_VALUES_ONLY does not accept "
            "MAX_SERIALIZED_SIZE"
        )
    endif()
    if(HAFS_PYTHON_VALUES_ONLY AND HAFS_DESCRIPTOR_TENSOR_CUSTOM_CODEC)
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema PYTHON_VALUES_ONLY does not generate a codec"
        )
    endif()
    if(HAFS_PYTHON AND HAFS_DESCRIPTOR_TENSOR_CUSTOM_CODEC AND
       NOT HAFS_PYTHON_LINK_LIBRARIES)
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema PYTHON with DESCRIPTOR_TENSOR_CUSTOM_CODEC "
            "requires PYTHON_LINK_LIBRARIES"
        )
    endif()
    if(HAFS_PYTHON AND HAFS_DESCRIPTOR_TENSOR_CUSTOM_CODEC AND
       NOT HAFS_PYTHON_TRAITS_INCLUDE)
        message(FATAL_ERROR
            "holoscan_add_flatbuffer_schema PYTHON with DESCRIPTOR_TENSOR_CUSTOM_CODEC "
            "requires PYTHON_TRAITS_INCLUDE"
        )
    endif()
    if((HAFS_PYTHON_CONSTANTS OR HAFS_PYTHON_EXTERNAL_TYPES OR HAFS_PYTHON_LINK_LIBRARIES OR
        HAFS_PYTHON_INSTALL_COMPONENT OR HAFS_PYTHON_OUTPUT_DIR OR
        HAFS_PYTHON_TRAITS_INCLUDE) AND
       NOT (HAFS_PYTHON OR HAFS_PYTHON_VALUES_ONLY))
        message(FATAL_ERROR
            "Python schema arguments require PYTHON or PYTHON_VALUES_ONLY"
        )
    endif()
    if(HAFS_PYTHON_TRAITS_INCLUDE AND NOT HAFS_PYTHON)
        message(FATAL_ERROR "PYTHON_TRAITS_INCLUDE requires PYTHON")
    endif()
    foreach(_holoscan_schema_traits_argument IN ITEMS
            HAFS_TRAITS_HEADER
            HAFS_ROOT_SCHEMA
            HAFS_COMPATIBILITY_EPOCH
            HAFS_NATIVE_BOUNDS_FILE)
        if(_holoscan_schema_traits_argument STREQUAL "HAFS_ROOT_SCHEMA" AND
           HAFS_PYTHON_VALUES_ONLY)
            continue()
        endif()
        if(DEFINED "${_holoscan_schema_traits_argument}"
           AND NOT "${${_holoscan_schema_traits_argument}}" STREQUAL ""
           AND NOT _holoscan_schema_traits_requested)
            string(REGEX REPLACE "^HAFS_" "" _holoscan_schema_argument_name
                "${_holoscan_schema_traits_argument}")
            message(FATAL_ERROR
                "${_holoscan_schema_argument_name} requires MAX_SERIALIZED_SIZE"
            )
        endif()
    endforeach()
    if(HAFS_PROVENANCE_FILE AND NOT _holoscan_schema_traits_requested)
        message(FATAL_ERROR "PROVENANCE_FILE requires MAX_SERIALIZED_SIZE")
    endif()
    if(HAFS_NATIVE_BOUNDS_FILE)
        if(HAFS_DESCRIPTOR_TENSOR_CUSTOM_CODEC)
            message(FATAL_ERROR "NATIVE_BOUNDS_FILE requires a generated codec, not a custom codec")
        endif()
        get_filename_component(_holoscan_native_bounds_file "${HAFS_NATIVE_BOUNDS_FILE}"
            ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        if(NOT EXISTS "${_holoscan_native_bounds_file}" OR IS_DIRECTORY "${_holoscan_native_bounds_file}")
            message(FATAL_ERROR "NATIVE_BOUNDS_FILE must name an existing JSON file")
        endif()
    endif()
    if((_holoscan_schema_traits_requested OR HAFS_PYTHON_VALUES_ONLY) AND
       NOT HAFS_ROOT_SCHEMA)
        list(LENGTH HAFS_SCHEMAS _holoscan_schema_count)
        if(NOT _holoscan_schema_count EQUAL 1)
            message(FATAL_ERROR
                "ROOT_SCHEMA is required when SCHEMAS contains more than one entry"
            )
        endif()
        list(GET HAFS_SCHEMAS 0 HAFS_ROOT_SCHEMA)
    endif()
    if(_holoscan_schema_traits_requested AND NOT HAFS_TRAITS_HEADER)
        get_filename_component(_holoscan_inferred_traits_stem "${HAFS_ROOT_SCHEMA}" NAME_WE)
        set(HAFS_TRAITS_HEADER "${_holoscan_inferred_traits_stem}_schema_traits.hpp")
    endif()
    if(HAFS_CONFORMANCE_SCHEMA AND NOT HAFS_CONFORMANCE_BASELINE)
        message(FATAL_ERROR "CONFORMANCE_SCHEMA requires CONFORMANCE_BASELINE")
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
    set(_holoscan_schema_traits_generator
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../scripts/generate_schema_traits.py"
    )
    if(NOT EXISTS "${_holoscan_schema_traits_generator}")
        set(_holoscan_schema_traits_generator
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/generate_schema_traits.py"
        )
    endif()
    if(NOT EXISTS "${_holoscan_schema_traits_generator}")
        message(FATAL_ERROR "Holoscan schema traits generator is missing")
    endif()

    if(NOT HAFS_OUTPUT_DIR)
        set(HAFS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/holoscan/schemas")
    endif()
    get_filename_component(
        _holoscan_schema_output_dir "${HAFS_OUTPUT_DIR}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
    )
    if(NOT HAFS_PYTHON_OUTPUT_DIR)
        set(HAFS_PYTHON_OUTPUT_DIR "${_holoscan_schema_output_dir}")
    endif()
    get_filename_component(
        _holoscan_schema_python_output_dir "${HAFS_PYTHON_OUTPUT_DIR}" ABSOLUTE
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

    if(_holoscan_schema_traits_requested OR HAFS_PYTHON_VALUES_ONLY)
        get_filename_component(
            _holoscan_schema_bfbs_input_abs "${HAFS_ROOT_SCHEMA}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        get_filename_component(
            _holoscan_schema_bfbs_project_root
            "${_holoscan_schema_bfbs_input_abs}" DIRECTORY
        )
        foreach(_holoscan_schema_include_dir IN LISTS _holoscan_schema_include_dirs_abs)
            file(RELATIVE_PATH _holoscan_schema_bfbs_relative
                "${_holoscan_schema_include_dir}" "${_holoscan_schema_bfbs_input_abs}")
            if(NOT IS_ABSOLUTE "${_holoscan_schema_bfbs_relative}" AND
               NOT _holoscan_schema_bfbs_relative MATCHES "^\\.\\.(/|$)")
                set(_holoscan_schema_bfbs_project_root "${_holoscan_schema_include_dir}")
                break()
            endif()
        endforeach()
    endif()

    # Applications select inputs and roots but do not reorder or override the SDK-pinned
    # object-API, reflection, or embedding flags.
    set(_holoscan_schema_flatc_flags)
    if(HAFS_BFBS_EMBED)
        list(APPEND _holoscan_schema_flatc_flags --bfbs-gen-embed --schema)
    endif()
    # Field IDs are the source-level compatibility authority for every generated table.
    list(APPEND _holoscan_schema_flatc_flags --require-explicit-ids)
    list(APPEND _holoscan_schema_flatc_flags
        --cpp
        --cpp-ptr-type std::shared_ptr
        --gen-object-api
        --gen-name-strings
    )
    if(HAFS_NATIVE_BOUNDS_FILE)
        list(APPEND _holoscan_schema_flatc_flags --bfbs-builtins)
    endif()
    set(_holoscan_schema_traits_header)
    set(_holoscan_schema_traits_commands)
    if(_holoscan_schema_traits_requested)
        if(IS_ABSOLUTE "${HAFS_TRAITS_HEADER}"
           OR HAFS_TRAITS_HEADER MATCHES "(^|/)\\.\\.(/|$)"
           OR HAFS_TRAITS_HEADER MATCHES "\\\\"
           OR NOT HAFS_TRAITS_HEADER MATCHES "\\.(h|hh|hpp|hxx)$")
            message(FATAL_ERROR
                "TRAITS_HEADER must be a relative header path below OUTPUT_DIR: "
                "${HAFS_TRAITS_HEADER}"
            )
        endif()
        if(NOT EXISTS "${_holoscan_flatbuffers_reflection_schema}")
            message(FATAL_ERROR
                "Schema traits generation requires the FlatBuffers reflection schema from "
                "the matching SDK FlatBuffers package"
            )
        endif()
        foreach(_holoscan_schema_positive_integer IN ITEMS
                HAFS_COMPATIBILITY_EPOCH
                HAFS_MAX_SERIALIZED_SIZE)
            if(_holoscan_schema_positive_integer STREQUAL
               "HAFS_COMPATIBILITY_EPOCH" AND
               "${HAFS_COMPATIBILITY_EPOCH}" STREQUAL "")
                continue()
            endif()
            if(NOT "${${_holoscan_schema_positive_integer}}" MATCHES "^[1-9][0-9]*$")
                string(REGEX REPLACE "^HAFS_" "" _holoscan_schema_argument_name
                    "${_holoscan_schema_positive_integer}")
                message(FATAL_ERROR
                    "${_holoscan_schema_argument_name} must be a positive integer"
                )
            endif()
        endforeach()
        if("${HAFS_COMPATIBILITY_EPOCH}" STREQUAL "")
            set(HAFS_COMPATIBILITY_EPOCH 1)
        endif()
        get_filename_component(
            _holoscan_schema_traits_input_abs "${HAFS_ROOT_SCHEMA}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        list(FIND _holoscan_schema_inputs_abs
            "${_holoscan_schema_traits_input_abs}" _holoscan_schema_traits_index
        )
        if(_holoscan_schema_traits_index EQUAL -1)
            message(FATAL_ERROR "ROOT_SCHEMA must name one SCHEMAS entry")
        endif()
        if(IS_ABSOLUTE "${HAFS_ROOT_SCHEMA}")
            message(FATAL_ERROR
                "ROOT_SCHEMA must use a canonical relative spelling so .bfbs filenames "
                "do not retain a host path"
            )
        endif()
        _holoscan_schema_validate_logical_path(
            "${HAFS_ROOT_SCHEMA}" "ROOT_SCHEMA"
        )
        list(GET _holoscan_schema_output_stems
            "${_holoscan_schema_traits_index}" _holoscan_schema_traits_stem
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
        if(HAFS_PYTHON)
            string(MAKE_C_IDENTIFIER
                "_${_holoscan_schema_traits_stem}_holoscan"
                _holoscan_schema_python_module_name
            )
            set(_holoscan_schema_python_file
                "${_holoscan_schema_python_output_dir}/${_holoscan_schema_traits_stem}.py"
            )
            set(_holoscan_schema_python_stub
                "${_holoscan_schema_python_output_dir}/${_holoscan_schema_traits_stem}.pyi"
            )
            set(_holoscan_schema_python_registration_source
                "${_holoscan_schema_traits_metadata_dir}/${_holoscan_schema_traits_stem}_python_registration.cpp"
            )
        endif()
        if(HAFS_PROVENANCE_FILE)
            get_filename_component(_holoscan_schema_provenance_file
                "${HAFS_PROVENANCE_FILE}" ABSOLUTE
                BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}"
            )
        endif()
        list(APPEND _holoscan_schema_generated_headers
            "${_holoscan_schema_traits_header}"
        )
        set(_holoscan_schema_traits_dependencies
            "${_holoscan_flatbuffers_reflection_schema}"
        )
        set(_holoscan_schema_traits_generator_args)
        if(HAFS_NATIVE_BOUNDS_FILE)
            list(APPEND _holoscan_schema_traits_dependencies "${_holoscan_native_bounds_file}")
            list(APPEND _holoscan_schema_traits_generator_args
                --native-bounds-file "${_holoscan_native_bounds_file}")
        endif()
        foreach(_holoscan_schema_flatc_flag IN LISTS _holoscan_schema_flatc_flags)
            list(APPEND _holoscan_schema_traits_generator_args
                "--flatc-flag=${_holoscan_schema_flatc_flag}"
            )
        endforeach()
        if(HAFS_DESCRIPTOR_TENSOR_CUSTOM_CODEC)
            list(APPEND _holoscan_schema_traits_generator_args
                --descriptor-tensor-custom-codec
            )
        endif()
        # Reflection bytes reach the generated binding only when this build embedded them, so
        # holoscan::binary_schema<T>() is available for exactly the roots that opted in.
        if(HAFS_BFBS_EMBED)
            list(APPEND _holoscan_schema_traits_generator_args
                --bfbs-embed-header "${_holoscan_schema_traits_stem}_bfbs_generated.h"
            )
        endif()
        if(HAFS_PYTHON)
            set(_holoscan_schema_python_traits_include "${HAFS_TRAITS_HEADER}")
            if(HAFS_PYTHON_TRAITS_INCLUDE)
                set(_holoscan_schema_python_traits_include "${HAFS_PYTHON_TRAITS_INCLUDE}")
            endif()
            list(APPEND _holoscan_schema_traits_generator_args
                --python-output "${_holoscan_schema_python_file}"
                --python-stub-output "${_holoscan_schema_python_stub}"
                --python-registration-output
                    "${_holoscan_schema_python_registration_source}"
                --python-module-name "${_holoscan_schema_python_module_name}"
                --traits-include "${_holoscan_schema_python_traits_include}"
            )
            foreach(_holoscan_schema_external_type IN LISTS HAFS_PYTHON_EXTERNAL_TYPES)
                list(APPEND _holoscan_schema_traits_generator_args
                    --python-external-type "${_holoscan_schema_external_type}"
                )
            endforeach()
            foreach(_holoscan_schema_python_constant IN LISTS HAFS_PYTHON_CONSTANTS)
                list(APPEND _holoscan_schema_traits_generator_args
                    --python-constant "${_holoscan_schema_python_constant}"
                )
            endforeach()
        endif()
        foreach(_holoscan_schema_include_dir IN LISTS _holoscan_schema_include_dirs_abs)
            list(APPEND _holoscan_schema_traits_generator_args
                --include-dir "${_holoscan_schema_include_dir}"
            )
        endforeach()
        if(HAFS_PROVENANCE_FILE)
            list(APPEND _holoscan_schema_traits_generator_args
                --provenance-output "${_holoscan_schema_provenance_file}"
            )
        endif()
        # Native metadata ownership qualification needs built-in customization attributes.
        set(_holoscan_schema_traits_commands
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${_holoscan_schema_traits_metadata_dir}"
            COMMAND "${_holoscan_schema_flatc_target}"
                --binary
                --schema
                --bfbs-builtins
                --bfbs-filenames "${_holoscan_schema_bfbs_project_root}"
                -o "${_holoscan_schema_traits_metadata_dir}"
                ${_holoscan_schema_include_args}
                "${HAFS_ROOT_SCHEMA}"
            COMMAND "${Python3_EXECUTABLE}"
                "${_holoscan_schema_traits_generator}"
                generate
                --flatc "$<TARGET_FILE:${_holoscan_schema_flatc_target}>"
                --expected-flatc-version "${_holoscan_schema_flatc_version}"
                --reflection-schema "${_holoscan_flatbuffers_reflection_schema}"
                --bfbs "${_holoscan_schema_traits_bfbs}"
                --output "${_holoscan_schema_traits_header}"
                --root-source "${_holoscan_schema_traits_input_abs}"
                --root-generated-header "${_holoscan_schema_traits_stem}_generated.h"
                --compatibility-epoch "${HAFS_COMPATIBILITY_EPOCH}"
                --max-serialized-size "${HAFS_MAX_SERIALIZED_SIZE}"
                ${_holoscan_schema_traits_generator_args}
        )
        set(_holoscan_schema_traits_byproducts "${_holoscan_schema_traits_bfbs}")
    endif()

    if(HAFS_PYTHON_VALUES_ONLY)
        list(LENGTH HAFS_SCHEMAS _holoscan_schema_value_schema_count)
        if(NOT _holoscan_schema_value_schema_count EQUAL 1)
            message(FATAL_ERROR
                "holoscan_add_flatbuffer_schema PYTHON_VALUES_ONLY requires one SCHEMA"
            )
        endif()
        get_filename_component(
            _holoscan_schema_value_input_abs "${HAFS_ROOT_SCHEMA}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        list(FIND _holoscan_schema_inputs_abs
            "${_holoscan_schema_value_input_abs}" _holoscan_schema_value_index
        )
        if(_holoscan_schema_value_index EQUAL -1)
            message(FATAL_ERROR "ROOT_SCHEMA must name one SCHEMAS entry")
        endif()
        list(GET _holoscan_schema_output_stems
            "${_holoscan_schema_value_index}" _holoscan_schema_value_stem
        )
        set(_holoscan_schema_python_file
            "${_holoscan_schema_python_output_dir}/${_holoscan_schema_value_stem}.py"
        )
        set(_holoscan_schema_python_stub
            "${_holoscan_schema_python_output_dir}/${_holoscan_schema_value_stem}.pyi"
        )
        set(_holoscan_schema_traits_metadata_dir
            "${CMAKE_CURRENT_BINARY_DIR}/${HAFS_TARGET}_schema_traits"
        )
        set(_holoscan_schema_traits_bfbs
            "${_holoscan_schema_traits_metadata_dir}/${_holoscan_schema_value_stem}.bfbs"
        )
        set(_holoscan_schema_value_generator_args)
        foreach(_holoscan_schema_flatc_flag IN LISTS _holoscan_schema_flatc_flags)
            list(APPEND _holoscan_schema_value_generator_args
                "--flatc-flag=${_holoscan_schema_flatc_flag}"
            )
        endforeach()
        foreach(_holoscan_schema_include_dir IN LISTS _holoscan_schema_include_dirs_abs)
            list(APPEND _holoscan_schema_value_generator_args
                --include-dir "${_holoscan_schema_include_dir}"
            )
        endforeach()
        foreach(_holoscan_schema_external_type IN LISTS HAFS_PYTHON_EXTERNAL_TYPES)
            list(APPEND _holoscan_schema_value_generator_args
                --python-external-type "${_holoscan_schema_external_type}"
            )
        endforeach()
        foreach(_holoscan_schema_python_constant IN LISTS HAFS_PYTHON_CONSTANTS)
            list(APPEND _holoscan_schema_value_generator_args
                --python-constant "${_holoscan_schema_python_constant}"
            )
        endforeach()
        set(_holoscan_schema_traits_commands
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${_holoscan_schema_traits_metadata_dir}"
            COMMAND "${_holoscan_schema_flatc_target}"
                --binary
                --schema
                --bfbs-builtins
                --bfbs-filenames "${_holoscan_schema_bfbs_project_root}"
                -o "${_holoscan_schema_traits_metadata_dir}"
                ${_holoscan_schema_include_args}
                "${HAFS_ROOT_SCHEMA}"
            COMMAND "${Python3_EXECUTABLE}"
                "${_holoscan_schema_traits_generator}"
                generate
                --flatc "$<TARGET_FILE:${_holoscan_schema_flatc_target}>"
                --expected-flatc-version "${_holoscan_schema_flatc_version}"
                --reflection-schema "${_holoscan_flatbuffers_reflection_schema}"
                --bfbs "${_holoscan_schema_traits_bfbs}"
                --root-source "${_holoscan_schema_value_input_abs}"
                --python-values-only
                --python-output "${_holoscan_schema_python_file}"
                --python-stub-output "${_holoscan_schema_python_stub}"
                ${_holoscan_schema_value_generator_args}
        )
        set(_holoscan_schema_traits_dependencies
            "${_holoscan_flatbuffers_reflection_schema}"
        )
        set(_holoscan_schema_traits_byproducts "${_holoscan_schema_traits_bfbs}")
    endif()

    set(_holoscan_schema_conformance_commands)
    set(_holoscan_schema_conformance_dependencies)
    if(HAFS_CONFORMANCE_BASELINE)
        get_filename_component(_holoscan_schema_conformance_baseline
            "${HAFS_CONFORMANCE_BASELINE}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        if(NOT EXISTS "${_holoscan_schema_conformance_baseline}"
           OR IS_DIRECTORY "${_holoscan_schema_conformance_baseline}"
           OR NOT _holoscan_schema_conformance_baseline MATCHES "\\.fbs$")
            message(FATAL_ERROR "CONFORMANCE_BASELINE must name an existing .fbs source")
        endif()
        get_filename_component(_holoscan_schema_conformance_baseline_dir
            "${_holoscan_schema_conformance_baseline}" DIRECTORY)
        if(HAFS_CONFORMANCE_SCHEMA)
            _holoscan_schema_validate_logical_path(
                "${HAFS_CONFORMANCE_SCHEMA}" "CONFORMANCE_SCHEMA"
            )
            get_filename_component(
                _holoscan_schema_conformance_candidate_abs
                "${HAFS_CONFORMANCE_SCHEMA}" ABSOLUTE
                BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
            )
            list(FIND _holoscan_schema_inputs_abs
                "${_holoscan_schema_conformance_candidate_abs}"
                _holoscan_schema_conformance_candidate_index
            )
            if(_holoscan_schema_conformance_candidate_index EQUAL -1)
                message(FATAL_ERROR "CONFORMANCE_SCHEMA must name one SCHEMAS entry")
            endif()
            set(_holoscan_schema_conformance_candidate "${HAFS_CONFORMANCE_SCHEMA}")
        else()
            list(LENGTH HAFS_SCHEMAS _holoscan_schema_conformance_schema_count)
            if(NOT _holoscan_schema_conformance_schema_count EQUAL 1)
                message(FATAL_ERROR
                    "CONFORMANCE_BASELINE requires CONFORMANCE_SCHEMA when SCHEMAS has multiple roots"
                )
            endif()
            list(GET HAFS_SCHEMAS 0 _holoscan_schema_conformance_candidate)
        endif()
        set(_holoscan_schema_conformance_commands
            COMMAND "${_holoscan_schema_flatc_target}"
                --require-explicit-ids
                --conform "${_holoscan_schema_conformance_baseline}"
                ${_holoscan_schema_include_args}
                "${_holoscan_schema_conformance_candidate}"
        )
        list(APPEND _holoscan_schema_conformance_dependencies
            "${_holoscan_schema_conformance_baseline}"
        )
    endif()

    set(_holoscan_schema_generated_outputs ${_holoscan_schema_generated_headers})
    if(HAFS_PYTHON OR HAFS_PYTHON_VALUES_ONLY)
        list(APPEND _holoscan_schema_generated_outputs
            "${_holoscan_schema_python_file}"
            "${_holoscan_schema_python_stub}"
        )
    endif()
    if(HAFS_PYTHON)
        list(APPEND _holoscan_schema_generated_outputs
            "${_holoscan_schema_python_registration_source}"
        )
    endif()

    set(_holoscan_schema_dependency_file
        "${CMAKE_CURRENT_BINARY_DIR}/${HAFS_TARGET}_flatbuffer_schema.d"
    )
    list(GET _holoscan_schema_generated_headers 0 _holoscan_schema_dependency_target)
    set(_holoscan_schema_dependency_args)
    foreach(_holoscan_schema_flatc_flag IN LISTS _holoscan_schema_flatc_flags)
        list(APPEND _holoscan_schema_dependency_args
            "--flatc-flag=${_holoscan_schema_flatc_flag}"
        )
    endforeach()
    foreach(_holoscan_schema_include_dir IN LISTS _holoscan_schema_include_dirs_abs)
        list(APPEND _holoscan_schema_dependency_args
            --include-dir "${_holoscan_schema_include_dir}"
        )
    endforeach()
    foreach(_holoscan_schema_input IN LISTS _holoscan_schema_inputs_abs)
        list(APPEND _holoscan_schema_dependency_args --schema "${_holoscan_schema_input}")
    endforeach()
    if(HAFS_CONFORMANCE_BASELINE)
        list(APPEND _holoscan_schema_dependency_args
            --include-dir "${_holoscan_schema_conformance_baseline_dir}"
            --schema "${_holoscan_schema_conformance_baseline}"
        )
    endif()

    add_custom_command(
        OUTPUT ${_holoscan_schema_generated_outputs} ${_holoscan_schema_provenance_file}
        BYPRODUCTS ${_holoscan_schema_traits_byproducts}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_holoscan_schema_output_dir}"
        COMMAND "${Python3_EXECUTABLE}"
            "${_holoscan_schema_traits_generator}"
            dependencies
            --flatc "$<TARGET_FILE:${_holoscan_schema_flatc_target}>"
            --output "${_holoscan_schema_dependency_file}"
            --target "${_holoscan_schema_dependency_target}"
            ${_holoscan_schema_dependency_args}
        COMMAND "${_holoscan_schema_flatc_target}"
            ${_holoscan_schema_flatc_flags}
            -o "${_holoscan_schema_output_dir}"
            ${_holoscan_schema_include_args}
            ${_holoscan_schema_inputs}
        ${_holoscan_schema_conformance_commands}
        ${_holoscan_schema_traits_commands}
        DEPENDS
            ${_holoscan_schema_flatc_target}
            ${_holoscan_schema_dependencies}
            ${_holoscan_schema_traits_dependencies}
            ${_holoscan_schema_conformance_dependencies}
            "${_holoscan_schema_traits_generator}"
        DEPFILE "${_holoscan_schema_dependency_file}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT
            "Generating ${HAFS_TARGET} with flatc ${_holoscan_schema_flatc_version}"
        VERBATIM
    )
    set(_holoscan_schema_generation_target "${HAFS_TARGET}__generate")
    add_custom_target("${_holoscan_schema_generation_target}"
        DEPENDS ${_holoscan_schema_generated_outputs} ${_holoscan_schema_provenance_file})
    # An interface library with private generated sources is both linkable and directly buildable.
    # This keeps the conventional `cmake --build --target <name>` workflow for code-generation-only
    # checks while allowing normal consumers to inherit its usage requirements.
    add_library("${HAFS_TARGET}" INTERFACE ${_holoscan_schema_generated_headers})
    add_dependencies("${HAFS_TARGET}" "${_holoscan_schema_generation_target}")
    target_include_directories("${HAFS_TARGET}" INTERFACE
        "$<BUILD_INTERFACE:${_holoscan_schema_output_dir}>"
    )
    target_link_libraries("${HAFS_TARGET}" INTERFACE flatbuffers::flatbuffers)
    if(TARGET holoscan::schema)
        target_link_libraries("${HAFS_TARGET}" INTERFACE holoscan::schema)
    endif()
    set_target_properties("${HAFS_TARGET}" PROPERTIES
        HOLOSCAN_FLATBUFFER_GENERATED_HEADERS "${_holoscan_schema_generated_headers}"
        HOLOSCAN_FLATBUFFER_OUTPUT_DIR "${_holoscan_schema_output_dir}"
        HOLOSCAN_FLATC_VERSION_CONFIGURED "${_holoscan_schema_flatc_version}"
        HOLOSCAN_FLATBUFFER_TRAITS_HEADER "${_holoscan_schema_traits_header}"
        HOLOSCAN_FLATBUFFER_GENERATION_TARGET "${_holoscan_schema_generation_target}"
        HOLOSCAN_FLATBUFFER_PROVENANCE_FILE "${_holoscan_schema_provenance_file}"
        HOLOSCAN_FLATBUFFER_PYTHON_MODULE "${_holoscan_schema_python_file}"
        HOLOSCAN_FLATBUFFER_PYTHON_STUB "${_holoscan_schema_python_stub}"
    )
    if(HAFS_PYTHON_INSTALL_COMPONENT)
        set_property(TARGET "${HAFS_TARGET}" PROPERTY
            HOLOSCAN_FLATBUFFER_PYTHON_INSTALL_COMPONENT
            "${HAFS_PYTHON_INSTALL_COMPONENT}")
        set_property(GLOBAL APPEND PROPERTY
            HOLOSCAN_PYTHON_SCHEMA_INSTALL_TARGETS "${HAFS_TARGET}")
    endif()
    if(HAFS_PYTHON)
        set(_holoscan_schema_python_target "${HAFS_TARGET}__python_codec")
        Python3_add_library("${_holoscan_schema_python_target}" MODULE USE_SABI 3.12 WITH_SOABI
            "${_holoscan_schema_python_registration_source}"
        )
        add_dependencies(
            "${_holoscan_schema_python_target}"
            "${_holoscan_schema_generation_target}"
        )
        target_compile_features("${_holoscan_schema_python_target}" PRIVATE cxx_std_20)
        target_include_directories("${_holoscan_schema_python_target}" PRIVATE
            "${_holoscan_schema_output_dir}"
            ${_holoscan_schema_include_dirs_abs}
        )
        target_link_libraries("${_holoscan_schema_python_target}" PRIVATE
            flatbuffers::flatbuffers
            holoscan::core
            Python3::SABIModule
            ${HAFS_PYTHON_LINK_LIBRARIES}
        )
        if(TARGET holoscan::schema)
            target_link_libraries("${_holoscan_schema_python_target}" PRIVATE holoscan::schema)
        endif()
        set_target_properties("${_holoscan_schema_python_target}" PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_VISIBILITY_PRESET hidden
            LIBRARY_OUTPUT_DIRECTORY "${_holoscan_schema_python_output_dir}"
            OUTPUT_NAME "${_holoscan_schema_python_module_name}"
            VISIBILITY_INLINES_HIDDEN YES
        )
        if(NOT HAFS_PYTHON_CODEC_DEFERRED)
            add_dependencies("${HAFS_TARGET}" "${_holoscan_schema_python_target}")
        endif()
        set_target_properties("${HAFS_TARGET}" PROPERTIES
            HOLOSCAN_FLATBUFFER_PYTHON_CODEC_TARGET
                "${_holoscan_schema_python_target}"
            HOLOSCAN_FLATBUFFER_PYTHON_CODEC_FILE
                "$<TARGET_FILE:${_holoscan_schema_python_target}>"
        )
    endif()
endfunction()
