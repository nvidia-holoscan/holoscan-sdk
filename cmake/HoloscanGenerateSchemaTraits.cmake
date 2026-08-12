# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

if(NOT DEFINED HST_CONFIG_FILE OR HST_CONFIG_FILE STREQUAL "")
    message(FATAL_ERROR "HST_CONFIG_FILE is required")
endif()
if(NOT EXISTS "${HST_CONFIG_FILE}")
    message(FATAL_ERROR "Schema traits configuration does not exist: ${HST_CONFIG_FILE}")
endif()
include("${HST_CONFIG_FILE}")

if(NOT DEFINED HST_FLATC_EXECUTABLE OR HST_FLATC_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "HST_FLATC_EXECUTABLE is required")
endif()
if(NOT EXISTS "${HST_FLATC_EXECUTABLE}")
    message(FATAL_ERROR "flatc does not exist: ${HST_FLATC_EXECUTABLE}")
endif()
if(NOT EXISTS "${HST_BFBS_FILE}")
    message(FATAL_ERROR "Generated binary schema does not exist: ${HST_BFBS_FILE}")
endif()
if(NOT EXISTS "${HST_BFBS_GENERATED_FILE}")
    message(FATAL_ERROR
        "Generated embedded binary schema header does not exist: ${HST_BFBS_GENERATED_FILE}"
    )
endif()
if(NOT DEFINED HST_FLATBUFFERS_REFLECTION_SCHEMA
   OR HST_FLATBUFFERS_REFLECTION_SCHEMA STREQUAL ""
   OR NOT EXISTS "${HST_FLATBUFFERS_REFLECTION_SCHEMA}")
    message(FATAL_ERROR
        "FlatBuffers reflection schema does not exist: "
        "${HST_FLATBUFFERS_REFLECTION_SCHEMA}"
    )
endif()

execute_process(
    COMMAND "${HST_FLATC_EXECUTABLE}" --version
    RESULT_VARIABLE flatc_version_result
    OUTPUT_VARIABLE flatc_build
    ERROR_VARIABLE flatc_version_stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
if(NOT flatc_version_result EQUAL 0)
    message(FATAL_ERROR
        "Could not query flatc version: ${flatc_version_stderr}"
    )
endif()
if(NOT flatc_build MATCHES "([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    message(FATAL_ERROR "flatc returned an unrecognized version: ${flatc_build}")
endif()
set(runtime_major "${CMAKE_MATCH_1}")
set(runtime_minor "${CMAKE_MATCH_2}")
set(runtime_revision "${CMAKE_MATCH_3}")
set(actual_flatc_version
    "${runtime_major}.${runtime_minor}.${runtime_revision}"
)
if(NOT actual_flatc_version VERSION_EQUAL "${HST_EXPECTED_FLATC_VERSION}")
    message(FATAL_ERROR
        "flatc version ${actual_flatc_version} does not match configured version "
        "${HST_EXPECTED_FLATC_VERSION}"
    )
endif()

# FlatBuffers documents reflection.fbs as the intermediate-representation schema for BFBS files:
# https://flatbuffers.dev/intermediate_representation/
# The flatc command contract separately documents schema-driven JSON output for binary inputs after
# `--`. Combining those two documented mechanisms decodes BFBS through its canonical meta-schema;
# upstream does not present this exact combined command as a separately named recipe.
#
# Inspect the complete reachable BFBS object graph. Ordinary generated object-API packing omits
# holoscan.schema.Tensor data by default or, under an explicit materialization scope, copies it into
# the FlatBuffer. Neither behavior satisfies the descriptor-capable Tensor transport contract. The
# traversal below follows reflection.fbs: Obj indices address Schema.objects, while Union indices
# address Schema.enums and each variant's union_type may address an Object. Refresh it together with
# the vendored reflection schema whenever the pinned FlatBuffers version changes.
get_filename_component(bfbs_stem "${HST_BFBS_FILE}" NAME_WE)
set(reflection_output_dir "${HST_BFBS_FILE}.reflection")
set(reflection_json "${reflection_output_dir}/${bfbs_stem}.json")
file(REMOVE_RECURSE "${reflection_output_dir}")
file(MAKE_DIRECTORY "${reflection_output_dir}")
execute_process(
    COMMAND "${HST_FLATC_EXECUTABLE}"
        --json --strict-json --defaults-json
        -o "${reflection_output_dir}"
        "${HST_FLATBUFFERS_REFLECTION_SCHEMA}"
        -- "${HST_BFBS_FILE}"
    RESULT_VARIABLE reflection_result
    OUTPUT_VARIABLE reflection_stdout
    ERROR_VARIABLE reflection_stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
if(NOT reflection_result EQUAL 0 OR NOT EXISTS "${reflection_json}")
    message(FATAL_ERROR
        "Could not reflect generated binary schema: ${reflection_stderr}"
    )
endif()
file(READ "${reflection_json}" reflected_schema)
file(REMOVE_RECURSE "${reflection_output_dir}")
string(JSON reflected_root_name ERROR_VARIABLE reflected_root_error
    GET "${reflected_schema}" root_table name
)
set(reflected_selected_root_name "${HST_ROOT_TYPE}")
string(REGEX REPLACE "^::" "" reflected_selected_root_name
    "${reflected_selected_root_name}"
)
string(REPLACE "::" "." reflected_selected_root_name
    "${reflected_selected_root_name}"
)
string(JSON reflected_object_count ERROR_VARIABLE reflected_objects_error
    LENGTH "${reflected_schema}" objects
)
string(JSON reflected_enum_count ERROR_VARIABLE reflected_enums_error
    LENGTH "${reflected_schema}" enums
)
if(reflected_root_error OR reflected_objects_error OR reflected_enums_error
   OR reflected_object_count EQUAL 0)
    message(FATAL_ERROR "Generated binary schema reflection is incomplete")
endif()

set(reachable_objects)
set(reflected_selected_root_found FALSE)
math(EXPR reflected_object_last "${reflected_object_count} - 1")
foreach(object_index RANGE 0 ${reflected_object_last})
    string(JSON object_name GET "${reflected_schema}" objects ${object_index} name)
    if("${object_name}" STREQUAL "${reflected_selected_root_name}")
        list(APPEND reachable_objects "${object_index}")
        set(reflected_selected_root_found TRUE)
        break()
    endif()
endforeach()
if(NOT reflected_selected_root_found)
    message(FATAL_ERROR
        "Generated binary schema selected traits root '${reflected_selected_root_name}' "
        "is absent from its object inventory"
    )
endif()

macro(hst_enqueue_reflected_type type_json)
    string(JSON reflected_base_type GET "${type_json}" base_type)
    string(JSON reflected_element_type GET "${type_json}" element)
    string(JSON reflected_type_index GET "${type_json}" index)
    set(reflected_object_reference FALSE)
    set(reflected_union_reference FALSE)
    if(reflected_base_type STREQUAL "Obj")
        set(reflected_object_reference TRUE)
    elseif(reflected_base_type MATCHES "^(Vector|Array|Vector64)$"
           AND reflected_element_type STREQUAL "Obj")
        set(reflected_object_reference TRUE)
    elseif(reflected_base_type STREQUAL "Union")
        set(reflected_union_reference TRUE)
    elseif(reflected_base_type MATCHES "^(Vector|Array|Vector64)$"
           AND reflected_element_type STREQUAL "Union")
        set(reflected_union_reference TRUE)
    endif()
    if(reflected_object_reference)
        if(reflected_type_index LESS 0
           OR reflected_type_index GREATER_EQUAL reflected_object_count)
            message(FATAL_ERROR "Generated binary schema has an invalid object reference")
        endif()
        list(APPEND reachable_objects "${reflected_type_index}")
    elseif(reflected_union_reference)
        if(reflected_type_index LESS 0
           OR reflected_type_index GREATER_EQUAL reflected_enum_count)
            message(FATAL_ERROR "Generated binary schema has an invalid union reference")
        endif()
        string(JSON reflected_union_value_count LENGTH
            "${reflected_schema}" enums ${reflected_type_index} values
        )
        if(reflected_union_value_count GREATER 0)
            math(EXPR reflected_union_value_last "${reflected_union_value_count} - 1")
            foreach(reflected_union_value_index RANGE 0 ${reflected_union_value_last})
                string(JSON reflected_union_type GET
                    "${reflected_schema}" enums ${reflected_type_index} values
                    ${reflected_union_value_index} union_type
                )
                string(JSON reflected_union_base_type GET
                    "${reflected_union_type}" base_type
                )
                if(reflected_union_base_type STREQUAL "Obj")
                    string(JSON reflected_union_object_index GET
                        "${reflected_union_type}" index
                    )
                    if(reflected_union_object_index LESS 0
                       OR reflected_union_object_index GREATER_EQUAL reflected_object_count)
                        message(FATAL_ERROR
                            "Generated binary schema has an invalid union object reference"
                        )
                    endif()
                    list(APPEND reachable_objects "${reflected_union_object_index}")
                endif()
            endforeach()
        endif()
    endif()
endmacro()

if(NOT HST_DESCRIPTOR_TENSOR)
    set(visited_objects)
    while(TRUE)
        list(LENGTH reachable_objects reachable_object_count)
        if(reachable_object_count EQUAL 0)
            break()
        endif()
        list(POP_FRONT reachable_objects object_index)
        if(object_index IN_LIST visited_objects)
            continue()
        endif()
        list(APPEND visited_objects "${object_index}")
        string(JSON object_name GET "${reflected_schema}" objects ${object_index} name)
        if(object_name STREQUAL "holoscan.schema.Tensor")
            message(FATAL_ERROR
                "HOLOSCAN_SCHEMA_TENSOR_REQUIRES_DESCRIPTOR_TRAITS: TRAITS_HEADER cannot generate "
                "ordinary owning traits for Tensor-bearing schema "
                "'${reflected_selected_root_name}'; use descriptor-backed Tensor schema traits"
            )
        endif()
        string(JSON reflected_field_count LENGTH
            "${reflected_schema}" objects ${object_index} fields
        )
        if(reflected_field_count GREATER 0)
            math(EXPR reflected_field_last "${reflected_field_count} - 1")
            foreach(reflected_field_index RANGE 0 ${reflected_field_last})
                string(JSON reflected_field_type GET
                    "${reflected_schema}" objects ${object_index} fields
                    ${reflected_field_index} type
                )
                hst_enqueue_reflected_type("${reflected_field_type}")
            endforeach()
        endif()
    endwhile()
endif()

function(hst_escape_cpp_string output value)
    string(REPLACE "\\" "\\\\" escaped "${value}")
    string(REPLACE "\"" "\\\"" escaped "${escaped}")
    string(REPLACE "\n" "\\n" escaped "${escaped}")
    string(REPLACE "\r" "\\r" escaped "${escaped}")
    set("${output}" "${escaped}" PARENT_SCOPE)
endfunction()

file(READ "${HST_BFBS_FILE}" bfbs_hex HEX)
string(TOLOWER "${bfbs_hex}" bfbs_hex)
file(READ "${HST_BFBS_GENERATED_FILE}" bfbs_generated_header)
string(REGEX MATCH
    "static const uint8_t bfbsData\\[[0-9]+\\] = \\{[^}]*\\}"
    bfbs_generated_array
    "${bfbs_generated_header}"
)
if(bfbs_generated_array STREQUAL "")
    message(FATAL_ERROR
        "Generated embedded binary schema header has no bfbsData array: "
        "${HST_BFBS_GENERATED_FILE}"
    )
endif()
string(REGEX MATCHALL "0[xX][0-9A-Fa-f][0-9A-Fa-f]" bfbs_generated_bytes
    "${bfbs_generated_array}")
string(REPLACE ";" "" bfbs_generated_hex "${bfbs_generated_bytes}")
string(REPLACE "0x" "" bfbs_generated_hex "${bfbs_generated_hex}")
string(REPLACE "0X" "" bfbs_generated_hex "${bfbs_generated_hex}")
string(TOLOWER "${bfbs_generated_hex}" bfbs_generated_hex)
if(NOT bfbs_generated_hex STREQUAL bfbs_hex)
    message(FATAL_ERROR
        "Generated embedded and reflected binary schemas differ: ${HST_BFBS_FILE}"
    )
endif()

string(REGEX REPLACE "(..)" "\\1:" bfbs_byte_aligned_hex "${bfbs_hex}")
foreach(host_path IN LISTS HST_HOST_PATHS)
    if(host_path STREQUAL "")
        continue()
    endif()
    file(TO_CMAKE_PATH "${host_path}" normalized_host_path)
    string(HEX "${normalized_host_path}" host_path_hex)
    string(TOLOWER "${host_path_hex}" host_path_hex)
    string(REGEX REPLACE "(..)" "\\1:" host_path_byte_aligned_hex "${host_path_hex}")
    string(FIND "${bfbs_byte_aligned_hex}" "${host_path_byte_aligned_hex}"
        bfbs_host_path_position)
    if(NOT bfbs_host_path_position EQUAL -1)
        message(FATAL_ERROR
            "Generated .bfbs retains host path: ${normalized_host_path}"
        )
    endif()
    foreach(public_value IN ITEMS
            "${HST_ROOT_LOGICAL_PATH}"
            "${HST_INCLUDE_ROOT_MAPPING}")
        string(FIND "${public_value}" "${normalized_host_path}" host_path_position)
        if(NOT host_path_position EQUAL -1)
            message(FATAL_ERROR
                "Host path survived schema evidence normalization: ${normalized_host_path}"
            )
        endif()
    endforeach()
    foreach(public_value IN LISTS HST_ORDERED_FLAGS HST_INCLUDE_LOGICAL_PATHS)
        string(FIND "${public_value}" "${normalized_host_path}" host_path_position)
        if(NOT host_path_position EQUAL -1)
            message(FATAL_ERROR
                "Host path survived schema evidence normalization: ${normalized_host_path}"
            )
        endif()
    endforeach()
endforeach()

list(LENGTH HST_INCLUDE_FILES include_count)
list(LENGTH HST_INCLUDE_LOGICAL_PATHS logical_include_count)
if(NOT include_count EQUAL logical_include_count)
    message(FATAL_ERROR
        "Schema include file and logical-path counts differ: "
        "${include_count} != ${logical_include_count}"
    )
endif()

file(SHA256 "${HST_BFBS_FILE}" bfbs_sha256)
list(LENGTH HST_ORDERED_FLAGS flag_count)

set(flag_entries)
foreach(flag IN LISTS HST_ORDERED_FLAGS)
    hst_escape_cpp_string(escaped_flag "${flag}")
    string(APPEND flag_entries "    \"${escaped_flag}\",\n")
endforeach()

set(include_entries)
if(include_count GREATER 0)
    math(EXPR include_last "${include_count} - 1")
    foreach(include_index RANGE 0 ${include_last})
        list(GET HST_INCLUDE_FILES ${include_index} include_file)
        list(GET HST_INCLUDE_LOGICAL_PATHS ${include_index} include_logical_path)
        if(NOT EXISTS "${include_file}")
            message(FATAL_ERROR "Schema include disappeared: ${include_file}")
        endif()
        file(SHA256 "${include_file}" include_sha256)
        hst_escape_cpp_string(escaped_include_path "${include_logical_path}")
        string(APPEND include_entries
            "    holoscan::SchemaPackageSource{\n"
            "        .logical_path = \"${escaped_include_path}\",\n"
            "        .content_sha256 = holoscan::schema::detail::schema_hex_bytes<32>(\n"
            "            \"${include_sha256}\"),\n"
            "    },\n"
        )
    endforeach()
endif()

foreach(variable IN ITEMS
        HST_ROOT_GENERATED_HEADER
        HST_BFBS_GENERATED_HEADER
        HST_ROOT_TYPE
        HST_NATIVE_TYPE
        HST_BINARY_SCHEMA_TYPE
        HST_PACKAGE_NAMESPACE
        HST_CANONICAL_NAME
        HST_ROOT_LOGICAL_PATH
        HST_INCLUDE_ROOT_MAPPING)
    hst_escape_cpp_string("${variable}_ESCAPED" "${${variable}}")
endforeach()
hst_escape_cpp_string(FLATC_BUILD_ESCAPED "${flatc_build}")

set(generator_format_version 1)
set(package_format_version 1)
if(HST_DESCRIPTOR_TENSOR)
    include("${CMAKE_CURRENT_LIST_DIR}/HoloscanGenerateDescriptorSchemaTraits.cmake")
else()
    string(CONFIGURE [=[
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated by holoscan_add_flatbuffer_schema(). Do not edit.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/schema/flatbuffer_schema_package.hpp>

#include "@HST_BFBS_GENERATED_HEADER_ESCAPED@"
#include "@HST_ROOT_GENERATED_HEADER_ESCAPED@"

namespace @HST_PACKAGE_NAMESPACE@ {

inline constexpr std::size_t kMaxSerializedSize = @HST_MAX_SERIALIZED_SIZE@U;
inline constexpr std::array<std::string_view, @flag_count@> kFlatcFlags{
@flag_entries@};
inline constexpr std::array<holoscan::SchemaPackageSource, @include_count@> kIncludes{
@include_entries@};

[[nodiscard]] consteval holoscan::SchemaIdentity make_identity() {
  holoscan::SchemaIdentity identity{
      .canonical_name = "@HST_CANONICAL_NAME_ESCAPED@",
      .version = @HST_SCHEMA_VERSION@U,
      .schema_sha256 = holoscan::schema::detail::schema_hex_bytes<32>(
          "@bfbs_sha256@"),
  };
  identity.id = holoscan::derive_schema_id(identity);
  return identity;
}

inline constexpr holoscan::SchemaIdentity kIdentity = make_identity();

}  // namespace @HST_PACKAGE_NAMESPACE@

namespace holoscan::schema {

template <>
struct FlatBufferSchemaTraits<@HST_ROOT_TYPE@> {
  using root_type = @HST_ROOT_TYPE@;
  static constexpr SchemaIdentity identity = @HST_PACKAGE_NAMESPACE@::kIdentity;
  static constexpr std::uint32_t schema_version = identity.version;
  static constexpr std::string_view root_name = identity.canonical_name;
  static constexpr std::size_t default_inline_capacity = @HST_INLINE_CAPACITY@U;
  static constexpr std::size_t default_descriptor_metadata_capacity =
      @HST_DESCRIPTOR_METADATA_CAPACITY@U;
};

}  // namespace holoscan::schema

namespace holoscan {

template <>
consteval SchemaIdentity schema_identity<@HST_ROOT_TYPE@>() {
  return @HST_PACKAGE_NAMESPACE@::kIdentity;
}

template <>
consteval SchemaIdentity schema_identity<@HST_NATIVE_TYPE@>() {
  return @HST_PACKAGE_NAMESPACE@::kIdentity;
}

template <>
struct GeneratedSchemaNativeTraits<@HST_NATIVE_TYPE@> {
  using root_type = @HST_ROOT_TYPE@;
  static constexpr std::size_t max_serialized_size =
      @HST_PACKAGE_NAMESPACE@::kMaxSerializedSize;
};

template <>
inline const SchemaPackageDescriptor<@HST_NATIVE_TYPE@>&
schema_package<@HST_NATIVE_TYPE@>() noexcept {
  static const SchemaPackageDescriptor<@HST_NATIVE_TYPE@> package =
      schema::detail::make_flatbuffer_native_package_from_static_evidence<
          @HST_NATIVE_TYPE@, @HST_ROOT_TYPE@, @HST_BINARY_SCHEMA_TYPE@>(
          schema_identity<@HST_NATIVE_TYPE@>(),
          SchemaPackageEvidence{
              .flatc_build = "@FLATC_BUILD_ESCAPED@",
              .ordered_flags = @HST_PACKAGE_NAMESPACE@::kFlatcFlags,
              .root_logical_path = "@HST_ROOT_LOGICAL_PATH_ESCAPED@",
              .includes = @HST_PACKAGE_NAMESPACE@::kIncludes,
              .include_root_mapping = "@HST_INCLUDE_ROOT_MAPPING_ESCAPED@",
              .host_paths_stripped = true,
              .generator_format_version = @generator_format_version@U,
              .package_format_version = @package_format_version@U,
              .runtime_abi = FlatBufferRuntimeABI{
                  .major = @runtime_major@U,
                  .minor = @runtime_minor@U,
                  .revision = @runtime_revision@U,
              },
          });
  return package;
}

}  // namespace holoscan
]=] generated_traits @ONLY)
endif()

get_filename_component(output_directory "${HST_OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
set(temporary_output "${HST_OUTPUT_FILE}.tmp")
file(WRITE "${temporary_output}" "${generated_traits}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${temporary_output}" "${HST_OUTPUT_FILE}"
    RESULT_VARIABLE copy_result
)
file(REMOVE "${temporary_output}")
if(NOT copy_result EQUAL 0)
    message(FATAL_ERROR "Could not publish generated schema traits: ${HST_OUTPUT_FILE}")
endif()
