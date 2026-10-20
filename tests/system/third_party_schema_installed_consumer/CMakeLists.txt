# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

project(holoscan_third_party_schema_installed_consumer LANGUAGES CXX)

if(NOT DEFINED HOLOSCAN_TEST_INSTALL_PREFIX OR HOLOSCAN_TEST_INSTALL_PREFIX STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_TEST_INSTALL_PREFIX must name the isolated SDK install prefix")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${HOLOSCAN_TEST_INSTALL_PREFIX}")
set(host_flatc "${HOLOSCAN_TEST_INSTALL_PREFIX}/bin/holoscan/flatc${CMAKE_EXECUTABLE_SUFFIX}")
add_executable(flatbuffers::flatc IMPORTED GLOBAL)
set_target_properties(flatbuffers::flatc PROPERTIES IMPORTED_LOCATION "${host_flatc}")
find_package(holoscan
    PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}"
    NO_DEFAULT_PATH
    REQUIRED
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
    NATIVE_BOUNDS_FILE native_bounds.json
    SCHEMAS third_party_common.fbs third_party_frame.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${generated_dir}"
    TRAITS_HEADER third_party_frame_traits.hpp
    ROOT_SCHEMA third_party_frame.fbs
    COMPATIBILITY_EPOCH 3
    MAX_SERIALIZED_SIZE 512
    BFBS_EMBED
)
# Both invocations embed, so the determinism comparison below also covers the reflection bytes.
set(generated_dir_repeat "${CMAKE_CURRENT_BINARY_DIR}/generated-b")
holoscan_add_flatbuffer_schema(
    TARGET third_party_frame_generated_repeat
    NATIVE_BOUNDS_FILE native_bounds.json
    SCHEMAS third_party_common.fbs third_party_frame.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${generated_dir_repeat}"
    TRAITS_HEADER third_party_frame_traits.hpp
    ROOT_SCHEMA third_party_frame.fbs
    COMPATIBILITY_EPOCH 3
    MAX_SERIALIZED_SIZE 512
    BFBS_EMBED
)
holoscan_add_flatbuffer_schema(
    TARGET third_party_tensor_schema_generated
    SCHEMA schema_tooling_tensor_probe.fbs
    OUTPUT_DIR "${generated_dir}"
)
holoscan_add_flatbuffer_schema(
    TARGET third_party_tensor_schema_inferred_descriptor_traits
    SCHEMA schema_tooling_tensor_probe.fbs
    MAX_SERIALIZED_SIZE 512
)
set_target_properties(third_party_tensor_schema_inferred_descriptor_traits PROPERTIES
    EXCLUDE_FROM_ALL TRUE)
holoscan_add_flatbuffer_schema(
    TARGET third_party_selected_tensor_schema_ordinary_traits
    SCHEMA schema_tooling_selected_tensor_probe.fbs
    MAX_SERIALIZED_SIZE 512
)
get_target_property(third_party_flatc_version_configured
    third_party_frame_generated HOLOSCAN_FLATC_VERSION_CONFIGURED)
get_target_property(installed_flatc_version holoscan::fbs_tools HOLOSCAN_FLATC_VERSION)
get_target_property(third_party_frame_generated_traits_header
    third_party_frame_generated HOLOSCAN_FLATBUFFER_TRAITS_HEADER)
get_target_property(third_party_frame_generated_repeat_traits_header
    third_party_frame_generated_repeat HOLOSCAN_FLATBUFFER_TRAITS_HEADER)
get_target_property(third_party_tensor_schema_inferred_descriptor_traits_header
    third_party_tensor_schema_inferred_descriptor_traits HOLOSCAN_FLATBUFFER_TRAITS_HEADER)
if(NOT third_party_flatc_version_configured OR NOT installed_flatc_version)
    message(FATAL_ERROR "Schema helper did not record the configured flatc version")
endif()
if(NOT third_party_flatc_version_configured STREQUAL installed_flatc_version)
    message(FATAL_ERROR "Schema helper did not record the configured flatc version")
endif()
if(NOT third_party_frame_generated_traits_header STREQUAL
       "${generated_dir}/third_party_frame_traits.hpp")
    message(FATAL_ERROR "Schema helper did not publish its generated traits header")
endif()

set(traits_determinism_stamp
    "${CMAKE_CURRENT_BINARY_DIR}/third_party_frame_traits.deterministic")
add_custom_command(
    OUTPUT "${traits_determinism_stamp}"
    COMMAND "${CMAKE_COMMAND}" -E compare_files
        "${third_party_frame_generated_traits_header}"
        "${third_party_frame_generated_repeat_traits_header}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${traits_determinism_stamp}"
    DEPENDS
        "${third_party_frame_generated_traits_header}"
        "${third_party_frame_generated_repeat_traits_header}"
    COMMENT "Comparing independently generated schema traits headers"
    VERBATIM
)
add_custom_target(third_party_frame_traits_determinism ALL
    DEPENDS "${traits_determinism_stamp}")

add_executable(third_party_schema_installed_consumer
    main.cpp
    ../../common/allocation_counter.cpp
)
add_dependencies(third_party_schema_installed_consumer
    third_party_frame_generated_repeat
    third_party_frame_traits_determinism
    third_party_tensor_schema_generated
)
target_compile_features(third_party_schema_installed_consumer PRIVATE cxx_std_20)
set_target_properties(third_party_schema_installed_consumer PROPERTIES CXX_EXTENSIONS OFF)
# The application consumes the FlatBuffers package configured by find_package(holoscan), including
# its bundled archive for parser APIs that are not header-only. holoscan::sensor_io is named because
# the consumer also reads the reflection bytes the SDK ships for its own Image and Imu payloads,
# which exercises the installed sensor-payload target rather than only Core.
target_link_libraries(third_party_schema_installed_consumer
    PRIVATE holoscan::core holoscan::sensor_io third_party_frame_generated)

# Keep the installed ImageDescriptorScratch ABI smoke check in this package consumer so it
# shares the Core/Sensor I/O installation and configuration.
add_executable(sensor_io_installed_consumer test_sensor_io.cpp)
target_compile_features(sensor_io_installed_consumer PRIVATE cxx_std_20)
target_link_libraries(sensor_io_installed_consumer PRIVATE holoscan::sensor_io)

include(CTest)
add_test(NAME sensor_io_installed_consumer.run COMMAND sensor_io_installed_consumer)
add_test(NAME third_party_schema_installed_consumer.run
    COMMAND third_party_schema_installed_consumer)
add_test(NAME third_party_schema_installed_consumer.tensor_traits_inferred
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_TEST_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "-DINFERRED_DESCRIPTOR_TRAITS_HEADER=${third_party_tensor_schema_inferred_descriptor_traits_header}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_tensor_traits_inference.cmake"
)
if(WIN32)
    set_tests_properties(third_party_schema_installed_consumer.run
        sensor_io_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/bin")
elseif(APPLE)
    set_tests_properties(third_party_schema_installed_consumer.run
        sensor_io_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "DYLD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
else()
    set_tests_properties(third_party_schema_installed_consumer.run
        sensor_io_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
endif()
