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
    SCHEMAS valid.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
)
set(common_traits_arguments
    BFBS_EMBED
    TRAITS_HEADER schema_tools_traits.hpp
    TRAITS_ROOT_TYPE schema_tools::Root
    TRAITS_NATIVE_TYPE schema_tools::RootT
    TRAITS_BINARY_SCHEMA_TYPE schema_tools::RootBinarySchema
    TRAITS_PACKAGE_NAMESPACE schema_tools::package
    TRAITS_CANONICAL_NAME schema_tools.Root
    TRAITS_SCHEMA_VERSION 1
    TRAITS_MAX_SERIALIZED_SIZE 128
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
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL inferred_traits_schema)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_inferred_traits_schema
        ${common_schema_arguments}
        ${common_traits_arguments}
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL explicit_capacities)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_explicit_capacities
        ${common_schema_arguments}
        ${common_traits_arguments}
        TRAITS_INLINE_CAPACITY 64
        TRAITS_DESCRIPTOR_METADATA_CAPACITY 96
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
        TRAITS_HEADER schema_tools_traits.hpp
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL invalid_type)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_invalid_type
        ${common_schema_arguments}
        BFBS_EMBED
        TRAITS_HEADER schema_tools_traits.hpp
        TRAITS_ROOT_TYPE Root
        TRAITS_NATIVE_TYPE schema_tools::RootT
        TRAITS_BINARY_SCHEMA_TYPE schema_tools::RootBinarySchema
        TRAITS_PACKAGE_NAMESPACE schema_tools::package
        TRAITS_CANONICAL_NAME schema_tools.Root
        TRAITS_SCHEMA_VERSION 1
        TRAITS_MAX_SERIALIZED_SIZE 128
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL invalid_capacity)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_invalid_capacity
        ${common_schema_arguments}
        BFBS_EMBED
        TRAITS_HEADER schema_tools_traits.hpp
        TRAITS_ROOT_TYPE schema_tools::Root
        TRAITS_NATIVE_TYPE schema_tools::RootT
        TRAITS_BINARY_SCHEMA_TYPE schema_tools::RootBinarySchema
        TRAITS_PACKAGE_NAMESPACE schema_tools::package
        TRAITS_CANONICAL_NAME schema_tools.Root
        TRAITS_SCHEMA_VERSION 1
        TRAITS_MAX_SERIALIZED_SIZE -1
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL zero_inline_capacity)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_zero_inline_capacity
        ${common_schema_arguments}
        ${common_traits_arguments}
        TRAITS_INLINE_CAPACITY 0
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL false_descriptor_capacity)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_false_descriptor_capacity
        ${common_schema_arguments}
        ${common_traits_arguments}
        TRAITS_DESCRIPTOR_METADATA_CAPACITY OFF
    )
elseif(HOLOSCAN_SCHEMA_TEST_CASE STREQUAL oversized_descriptor_capacity)
    holoscan_add_flatbuffer_schema(
        TARGET schema_tools_oversized_descriptor_capacity
        ${common_schema_arguments}
        ${common_traits_arguments}
        DESCRIPTOR_TENSOR
        TRAITS_DESCRIPTOR_METADATA_CAPACITY 65537
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
