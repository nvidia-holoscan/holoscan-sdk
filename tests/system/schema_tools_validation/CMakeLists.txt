# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)
project(holoscan_schema_tools_validation LANGUAGES NONE)

foreach(required_variable IN ITEMS
        HOLOSCAN_SCHEMA_TOOLS_MODULE
        HOLOSCAN_FLATC_EXECUTABLE
        HOLOSCAN_FLATC_VERSION
        HOLOSCAN_FLATBUFFERS_REFLECTION_SCHEMA
        HOLOSCAN_SCHEMA_TEST_CASE)
    if(NOT DEFINED "${required_variable}" OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

add_executable(flatbuffers::flatc IMPORTED GLOBAL)
set_target_properties(flatbuffers::flatc PROPERTIES
    IMPORTED_LOCATION "${HOLOSCAN_FLATC_EXECUTABLE}"
)
add_library(holoscan::fbs_tools INTERFACE IMPORTED GLOBAL)
set_target_properties(holoscan::fbs_tools PROPERTIES
    HOLOSCAN_FLATC_VERSION "${HOLOSCAN_FLATC_VERSION}"
    HOLOSCAN_SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    HOLOSCAN_FLATBUFFERS_REFLECTION_SCHEMA "${HOLOSCAN_FLATBUFFERS_REFLECTION_SCHEMA}"
)
include("${HOLOSCAN_SCHEMA_TOOLS_MODULE}")

set(common_schema_arguments
    SCHEMA valid.fbs
)
set(common_traits_arguments
    MAX_SERIALIZED_SIZE 128
)

if(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL commented_include)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_commented_include
        SCHEMAS commented_include.fbs
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL false_logical_path)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_false_logical_path
        SCHEMAS false_logical_path.fbs
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL non_embed)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_non_embed
        ${common_schema_arguments}
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL embed)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_embed
        ${common_schema_arguments}
        ${common_traits_arguments}
        BFBS_EMBED
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL inferred_traits_schema)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_inferred_traits_schema
        ${common_schema_arguments}
        ${common_traits_arguments}
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL conformance)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_conformance
        ${common_schema_arguments}
        ${common_traits_arguments}
        CONFORMANCE_BASELINE valid.fbs
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL provenance)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_provenance
        ${common_schema_arguments}
        ${common_traits_arguments}
        PROVENANCE_FILE schema.provenance.json
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL incompatible_conformance)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_incompatible_conformance
        SCHEMAS incompatible_candidate.fbs
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
        CONFORMANCE_BASELINE valid.fbs
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL multi_root_conformance)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_multi_root_conformance
        SCHEMAS valid.fbs other.fbs
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
        CONFORMANCE_SCHEMA valid.fbs
        CONFORMANCE_BASELINE valid.fbs
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL ambiguous_conformance_schema)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_ambiguous_conformance_schema
        SCHEMAS valid.fbs other.fbs
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
        CONFORMANCE_BASELINE valid.fbs
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL nonmember_conformance_schema)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_nonmember_conformance_schema
        SCHEMAS valid.fbs other.fbs
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
        CONFORMANCE_SCHEMA incompatible_candidate.fbs
        CONFORMANCE_BASELINE valid.fbs
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL implicit_ids)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_implicit_ids
        SCHEMAS implicit_ids.fbs
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL unknown_argument)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_unknown_argument
        UNKNOWN_ARGUMENT rejected
        SCHEMAS valid.fbs
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL missing_target)
    holoscan_add_flatbuffer_schema(${common_schema_arguments})
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL missing_schemas)
    holoscan_add_flatbuffer_schema(TARGET schema_tools_missing_schemas)
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL missing_include_directory)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_missing_include_directory
        SCHEMAS valid.fbs
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/missing"
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL unresolved_include)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_unresolved_include
        SCHEMAS unresolved_include.fbs
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL traits_without_embed)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_traits_without_embed
        ${common_schema_arguments}
        MAX_SERIALIZED_SIZE 128
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL invalid_capacity)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_invalid_capacity
        ${common_schema_arguments}
        MAX_SERIALIZED_SIZE -1
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL ambiguous_traits_schema)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_ambiguous_traits_schema
        SCHEMAS valid.fbs other.fbs
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
        ${common_traits_arguments}
    )
else()
    message(FATAL_ERROR "Unknown HOLOSCAN_SCHEMA_TEST_CASE=${HOLOSCAN_SCHEMA_TEST_CASE}")
endif()
