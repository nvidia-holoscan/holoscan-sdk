# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

project(holoscan_third_party_schema_installed_consumer LANGUAGES CXX)

if(NOT DEFINED HOLOSCAN_TEST_INSTALL_PREFIX OR HOLOSCAN_TEST_INSTALL_PREFIX STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_TEST_INSTALL_PREFIX must name the isolated SDK install prefix")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${HOLOSCAN_TEST_INSTALL_PREFIX}")
set(host_flatc "${HOLOSCAN_TEST_INSTALL_PREFIX}/bin/flatc${CMAKE_EXECUTABLE_SUFFIX}")
add_executable(flatbuffers::flatc IMPORTED GLOBAL)
set_target_properties(flatbuffers::flatc PROPERTIES IMPORTED_LOCATION "${host_flatc}")
find_package(holoscan REQUIRED CONFIG
    PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}"
    NO_DEFAULT_PATH
)

if(NOT TARGET holoscan::core)
    message(FATAL_ERROR "Installed package did not define holoscan::core")
endif()
if(NOT TARGET holoscan::schema)
    message(FATAL_ERROR "Installed package did not define holoscan::schema")
endif()
if(NOT TARGET flatbuffers::flatbuffers)
    message(FATAL_ERROR "Installed Holoscan package did not provide flatbuffers::flatbuffers")
endif()
if(NOT DEFINED holoscan_FLATBUFFERS_VERSION OR
   NOT holoscan_FLATBUFFERS_VERSION VERSION_EQUAL "25.12.19")
    message(FATAL_ERROR "Installed Holoscan package did not expose FlatBuffers version 25.12.19")
endif()
get_target_property(third_party_flatbuffers_imported flatbuffers::flatbuffers IMPORTED)
get_target_property(third_party_flatbuffers_type flatbuffers::flatbuffers TYPE)
get_target_property(third_party_flatbuffers_configs
    flatbuffers::flatbuffers IMPORTED_CONFIGURATIONS)
if(NOT third_party_flatbuffers_imported OR
   NOT third_party_flatbuffers_type STREQUAL "STATIC_LIBRARY" OR
   NOT third_party_flatbuffers_configs)
    message(FATAL_ERROR "Installed FlatBuffers target is not an imported static library")
endif()
foreach(third_party_flatbuffers_config IN LISTS third_party_flatbuffers_configs)
    string(TOUPPER "${third_party_flatbuffers_config}" third_party_flatbuffers_config_upper)
    get_target_property(third_party_flatbuffers_location flatbuffers::flatbuffers
        "IMPORTED_LOCATION_${third_party_flatbuffers_config_upper}")
    if(third_party_flatbuffers_location)
        break()
    endif()
endforeach()
get_filename_component(third_party_flatbuffers_filename
    "${third_party_flatbuffers_location}" NAME)
if(NOT third_party_flatbuffers_location OR
   NOT third_party_flatbuffers_filename STREQUAL
       "${CMAKE_STATIC_LIBRARY_PREFIX}flatbuffers${CMAKE_STATIC_LIBRARY_SUFFIX}")
    message(FATAL_ERROR "Installed FlatBuffers target does not resolve to its static archive")
endif()
get_target_property(configured_flatc flatbuffers::flatc IMPORTED_LOCATION)
if(NOT configured_flatc STREQUAL host_flatc)
    message(FATAL_ERROR "Installed package replaced the predeclared host flatc executable")
endif()

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated-a")
holoscan_add_flatbuffer_schema(
    TARGET third_party_frame_generated
    SCHEMAS third_party_common.fbs third_party_frame.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${generated_dir}"
    BFBS_EMBED
    TRAITS_HEADER third_party_frame_traits.hpp
    TRAITS_SCHEMA third_party_frame.fbs
    TRAITS_ROOT_TYPE app3p::Frame
    TRAITS_NATIVE_TYPE app3p::FrameT
    TRAITS_BINARY_SCHEMA_TYPE app3p::FrameBinarySchema
    TRAITS_PACKAGE_NAMESPACE app3p::package
    TRAITS_CANONICAL_NAME app3p.Frame
    TRAITS_SCHEMA_VERSION 1
    TRAITS_MAX_SERIALIZED_SIZE 512
)
set(generated_dir_repeat "${CMAKE_CURRENT_BINARY_DIR}/generated-b")
holoscan_add_flatbuffer_schema(
    TARGET third_party_frame_generated_repeat
    SCHEMAS third_party_common.fbs third_party_frame.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${generated_dir_repeat}"
    BFBS_EMBED
    TRAITS_HEADER third_party_frame_traits.hpp
    TRAITS_SCHEMA third_party_frame.fbs
    TRAITS_ROOT_TYPE app3p::Frame
    TRAITS_NATIVE_TYPE app3p::FrameT
    TRAITS_BINARY_SCHEMA_TYPE app3p::FrameBinarySchema
    TRAITS_PACKAGE_NAMESPACE app3p::package
    TRAITS_CANONICAL_NAME app3p.Frame
    TRAITS_SCHEMA_VERSION 1
    TRAITS_MAX_SERIALIZED_SIZE 512
)
holoscan_add_flatbuffer_schema(
    TARGET third_party_tensor_schema_generated
    SCHEMAS schema_tooling_tensor_probe.fbs
    OUTPUT_DIR "${generated_dir}"
    BFBS_EMBED
)
holoscan_add_flatbuffer_schema(
    TARGET third_party_tensor_schema_ordinary_traits_rejected
    SCHEMAS schema_tooling_tensor_probe.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated-tensor-rejected"
    BFBS_EMBED
    TRAITS_HEADER schema_tooling_tensor_probe_traits.hpp
    TRAITS_ROOT_TYPE holoscan::schema::SchemaToolingTensorProbe
    TRAITS_NATIVE_TYPE holoscan::schema::SchemaToolingTensorProbeT
    TRAITS_BINARY_SCHEMA_TYPE holoscan::schema::SchemaToolingTensorProbeBinarySchema
    TRAITS_PACKAGE_NAMESPACE holoscan::schema::tensor_probe_package
    TRAITS_CANONICAL_NAME holoscan.schema.SchemaToolingTensorProbe
    TRAITS_SCHEMA_VERSION 1
    TRAITS_MAX_SERIALIZED_SIZE 512
)
holoscan_add_flatbuffer_schema(
    TARGET third_party_selected_tensor_schema_ordinary_traits_rejected
    SCHEMAS schema_tooling_selected_tensor_probe.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated-selected-tensor-rejected"
    BFBS_EMBED
    TRAITS_HEADER schema_tooling_selected_tensor_probe_traits.hpp
    TRAITS_ROOT_TYPE holoscan::schema::SchemaToolingSelectedTensorProbe
    TRAITS_NATIVE_TYPE holoscan::schema::SchemaToolingSelectedTensorProbeT
    TRAITS_BINARY_SCHEMA_TYPE holoscan::schema::SchemaToolingSelectedTensorProbeBinarySchema
    TRAITS_PACKAGE_NAMESPACE holoscan::schema::selected_tensor_probe_package
    TRAITS_CANONICAL_NAME holoscan.schema.SchemaToolingSelectedTensorProbe
    TRAITS_SCHEMA_VERSION 1
    TRAITS_MAX_SERIALIZED_SIZE 512
)
get_target_property(third_party_flatc_version_configured
    third_party_frame_generated HOLOSCAN_FLATC_VERSION_CONFIGURED)
get_target_property(installed_flatc_version holoscan::fbs_tools HOLOSCAN_FLATC_VERSION)
if(NOT third_party_flatc_version_configured OR NOT installed_flatc_version)
    message(FATAL_ERROR "Schema helper did not record the configured flatc version")
endif()
if(NOT third_party_flatc_version_configured STREQUAL installed_flatc_version)
    message(FATAL_ERROR "Schema helper did not record the configured flatc version")
endif()
if(NOT third_party_frame_generated_TRAITS_HEADER STREQUAL
       "${generated_dir}/third_party_frame_traits.hpp")
    message(FATAL_ERROR "Schema helper did not publish its generated traits header")
endif()

set(traits_determinism_stamp
    "${CMAKE_CURRENT_BINARY_DIR}/third_party_frame_traits.deterministic")
add_custom_command(
    OUTPUT "${traits_determinism_stamp}"
    COMMAND "${CMAKE_COMMAND}" -E compare_files
        "${third_party_frame_generated_TRAITS_HEADER}"
        "${third_party_frame_generated_repeat_TRAITS_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${traits_determinism_stamp}"
    DEPENDS
        "${third_party_frame_generated_TRAITS_HEADER}"
        "${third_party_frame_generated_repeat_TRAITS_HEADER}"
    COMMENT "Comparing independently generated schema traits headers"
    VERBATIM
)
add_custom_target(third_party_frame_traits_determinism ALL
    DEPENDS "${traits_determinism_stamp}")

add_executable(third_party_schema_installed_consumer main.cpp)
add_dependencies(third_party_schema_installed_consumer
    third_party_frame_generated
    third_party_frame_generated_repeat
    third_party_frame_traits_determinism
    third_party_tensor_schema_generated
)
target_compile_features(third_party_schema_installed_consumer PRIVATE cxx_std_20)
target_include_directories(third_party_schema_installed_consumer
    PRIVATE "${generated_dir}")
set_target_properties(third_party_schema_installed_consumer PROPERTIES CXX_EXTENSIONS OFF)
# The application consumes the FlatBuffers package configured by find_package(holoscan), including
# its bundled archive for parser APIs that are not header-only.
target_link_libraries(third_party_schema_installed_consumer
    PRIVATE holoscan::core holoscan::schema flatbuffers::flatbuffers)

include(CTest)
add_test(NAME third_party_schema_installed_consumer.run
    COMMAND third_party_schema_installed_consumer)
add_test(NAME third_party_schema_installed_consumer.tensor_traits_rejected
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_TEST_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/expect_tensor_traits_rejection.cmake"
)
if(WIN32)
    set_tests_properties(third_party_schema_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/bin")
elseif(APPLE)
    set_tests_properties(third_party_schema_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "DYLD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
else()
    set_tests_properties(third_party_schema_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
endif()
