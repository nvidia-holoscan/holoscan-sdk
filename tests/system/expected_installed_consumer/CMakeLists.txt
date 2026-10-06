# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_expected_installed_consumer LANGUAGES CXX)

if(NOT DEFINED HOLOSCAN_TEST_INSTALL_PREFIX OR HOLOSCAN_TEST_INSTALL_PREFIX STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_TEST_INSTALL_PREFIX must name the isolated SDK install prefix")
endif()

# Keep every transitive find_dependency() lookup inside the isolated SDK prefix. CUDAToolkit is
# intentionally supplied as an explicit external toolchain root by the parent package test; a
# missing dependency that the SDK promises to bundle must not be masked by a developer package,
# environment path, package registry, or system installation.
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY TRUE)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY TRUE)
set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)
set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)
set(CMAKE_FIND_USE_INSTALL_PREFIX FALSE)
list(PREPEND CMAKE_PREFIX_PATH "${HOLOSCAN_TEST_INSTALL_PREFIX}")

# NO_DEFAULT_PATH additionally prevents the Holoscan package itself from resolving outside the
# isolated prefix.
find_package(holoscan REQUIRED CONFIG
    PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}"
    NO_DEFAULT_PATH
)

foreach(imported_target IN ITEMS holoscan::core holoscan::logger)
    if(NOT TARGET ${imported_target})
        message(FATAL_ERROR "Installed package did not define ${imported_target}")
    endif()
    get_target_property(imported_definitions ${imported_target} INTERFACE_COMPILE_DEFINITIONS)
    if(imported_definitions AND imported_definitions MATCHES "HOLOSCAN_LOG_ACTIVE_LEVEL")
        message(FATAL_ERROR
            "${imported_target} leaked the SDK build-time HOLOSCAN_LOG_ACTIVE_LEVEL policy: "
            "${imported_definitions}"
        )
    endif()
endforeach()

# Exercise one policy state through two separately compiled consumers of the same Core instance.
# Static Core uses a static peer: loading independent static runtimes into separate DSOs is outside
# the process-wide policy contract.
get_target_property(core_library_type holoscan::core TYPE)
if(core_library_type STREQUAL "SHARED_LIBRARY")
    add_library(diagnostic_policy_peer SHARED diagnostic_policy_peer.cpp)
else()
    add_library(diagnostic_policy_peer STATIC diagnostic_policy_peer.cpp)
endif()
include(GenerateExportHeader)
generate_export_header(diagnostic_policy_peer)
target_include_directories(diagnostic_policy_peer PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
if(NOT core_library_type STREQUAL "SHARED_LIBRARY")
    target_compile_definitions(diagnostic_policy_peer PUBLIC DIAGNOSTIC_POLICY_PEER_STATIC_DEFINE)
endif()
target_compile_features(diagnostic_policy_peer PRIVATE cxx_std_20)
target_link_libraries(diagnostic_policy_peer PRIVATE holoscan::core)

add_executable(expected_installed_consumer main.cpp)
target_compile_features(expected_installed_consumer PRIVATE cxx_std_20)
set_target_properties(expected_installed_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(expected_installed_consumer
    PRIVATE
        holoscan::core
        holoscan::logger
        diagnostic_policy_peer
)

# Compile the header-isolation translation unit a second time through the installed package. It
# includes expected_macros.hpp as its only header, so this target proves install-tree
# self-containment rather than relying on the build-tree smoke target.
add_executable(expected_macros_installed_consumer ../expected_macros_header_smoke.cpp)
target_compile_features(expected_macros_installed_consumer PRIVATE cxx_std_20)
set_target_properties(expected_macros_installed_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(expected_macros_installed_consumer PRIVATE holoscan::core)

# Include the installed MemoryBuffer header alone and exercise caller-side ERROR compilation.
add_executable(memory_buffer_header_smoke memory_buffer_header_smoke.cpp)
target_compile_features(memory_buffer_header_smoke PRIVATE cxx_std_20)
target_compile_definitions(memory_buffer_header_smoke PRIVATE
    HOLOSCAN_LOG_ACTIVE_LEVEL=HOLOSCAN_LOG_LEVEL_OFF)
target_link_libraries(memory_buffer_header_smoke PRIVATE holoscan::core)

include(CTest)
add_test(NAME expected_installed_consumer.run COMMAND expected_installed_consumer)
set_tests_properties(expected_installed_consumer.run PROPERTIES
    ENVIRONMENT "HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING="
)
add_test(NAME expected_macros_installed_consumer.run COMMAND expected_macros_installed_consumer)
add_test(NAME memory_buffer_header_smoke.run COMMAND memory_buffer_header_smoke)
set_tests_properties(memory_buffer_header_smoke.run PROPERTIES
    ENVIRONMENT "HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING=ON;HOLOSCAN_LOG_LEVEL=ERROR;HOLOSCAN_LOG_FORMAT=%v"
    FAIL_REGULAR_EXPRESSION "Native allocator failed")
# The executable sets the opposite API preference; the peer DSO reports through the same Core.
add_test(NAME expected_installed_consumer.memory_buffer_off COMMAND expected_installed_consumer ON)
set_tests_properties(expected_installed_consumer.memory_buffer_off PROPERTIES
    ENVIRONMENT "HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING=OFF;HOLOSCAN_LOG_LEVEL=ERROR;HOLOSCAN_LOG_FORMAT=%v"
    FAIL_REGULAR_EXPRESSION "Native allocator failed|Failed to free memory")
add_test(NAME expected_installed_consumer.memory_buffer_on
    COMMAND ${CMAKE_COMMAND} "-DCONSUMER=$<TARGET_FILE:expected_installed_consumer>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/check_memory_reporting.cmake")
set_tests_properties(expected_installed_consumer.memory_buffer_on PROPERTIES
    ENVIRONMENT "HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING=ON;HOLOSCAN_LOG_LEVEL=ERROR;HOLOSCAN_LOG_FORMAT=%v")
if(WIN32)
    set_tests_properties(expected_installed_consumer.run expected_macros_installed_consumer.run
        expected_installed_consumer.memory_buffer_on expected_installed_consumer.memory_buffer_off
        memory_buffer_header_smoke.run
        PROPERTIES
        ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/bin"
    )
elseif(APPLE)
    set_tests_properties(expected_installed_consumer.run expected_macros_installed_consumer.run
        expected_installed_consumer.memory_buffer_on expected_installed_consumer.memory_buffer_off
        memory_buffer_header_smoke.run
        PROPERTIES
        ENVIRONMENT_MODIFICATION
            "DYLD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib"
    )
else()
    set_tests_properties(expected_installed_consumer.run expected_macros_installed_consumer.run
        expected_installed_consumer.memory_buffer_on expected_installed_consumer.memory_buffer_off
        memory_buffer_header_smoke.run
        PROPERTIES
        ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib"
    )
endif()
