# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_target_configuration_test LANGUAGES CXX)

if(NOT DEFINED HOLOSCAN_MODULE_DIR OR NOT DEFINED HOLOSCAN_SOURCE_DIR)
    message(FATAL_ERROR "HOLOSCAN_MODULE_DIR and HOLOSCAN_SOURCE_DIR are required")
endif()
list(APPEND CMAKE_MODULE_PATH "${HOLOSCAN_MODULE_DIR}")
include(HoloscanTargetConfiguration)

# Simulate CMake's clang-cl classification after the host compiler has initialized the test
# project. The helpers only inspect these classification variables; the probe target is not built.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_CXX_COMPILER_ID Clang)
set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT MSVC)
set(MSVC TRUE)
set(BUILD_SHARED_LIBS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/custom-lib")
unset(CMAKE_BUILD_RPATH)
unset(CMAKE_INSTALL_RPATH)

holoscan_platform_has_native_descriptor_readiness(_windows_has_descriptor_readiness)
if(_windows_has_descriptor_readiness)
    message(FATAL_ERROR "Windows was classified as supporting Linux descriptor readiness")
endif()
holoscan_select_native_descriptor_readiness_source(
    _windows_readiness_source readiness_handle_linux.cpp readiness_handle_unsupported.cpp)
if(NOT _windows_readiness_source STREQUAL "readiness_handle_unsupported.cpp")
    message(FATAL_ERROR "Windows did not select the unsupported readiness implementation")
endif()

holoscan_is_linux_gnu_like_frontend(_clang_cl_is_linux_gnu_like)
if(_clang_cl_is_linux_gnu_like)
    message(FATAL_ERROR "clang-cl was classified as a Linux GNU-like frontend")
endif()
holoscan_configure_linux_origin_rpath()
if(CMAKE_BUILD_RPATH OR CMAKE_INSTALL_RPATH)
    message(FATAL_ERROR "clang-cl probe received an ELF origin-relative RPATH")
endif()

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp" "void probe() {}\n")
add_library(portability_probe SHARED "${CMAKE_CURRENT_BINARY_DIR}/probe.cpp")
add_library(archive_probe INTERFACE)
holoscan_configure_cxx20_target(portability_probe PRIVATE)
holoscan_configure_linux_shared_link(portability_probe)
holoscan_configure_linux_archive_hiding(archive_probe libprobe)
holoscan_apply_logging_policy(portability_probe)

get_target_property(_compile_features portability_probe COMPILE_FEATURES)
if(NOT "cxx_std_20" IN_LIST _compile_features)
    message(FATAL_ERROR "clang-cl probe did not receive the C++20 compile feature")
endif()
get_target_property(_compile_options portability_probe COMPILE_OPTIONS)
set(_expected_msvc_conformance_options
    "$<$<COMPILE_LANGUAGE:CXX>:/permissive->"
    "$<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>"
    "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
)
foreach(_expected_option IN LISTS _expected_msvc_conformance_options)
    if(NOT _expected_option IN_LIST _compile_options)
        message(FATAL_ERROR "clang-cl probe is missing conformance option ${_expected_option}")
    endif()
endforeach()

get_target_property(_link_options portability_probe LINK_OPTIONS)
if(_link_options)
    message(FATAL_ERROR "clang-cl probe received Linux/GNU linker options: ${_link_options}")
endif()
get_target_property(_archive_link_options archive_probe INTERFACE_LINK_OPTIONS)
if(_archive_link_options)
    message(FATAL_ERROR "clang-cl probe received Linux archive-hiding options")
endif()

get_target_property(_compile_definitions portability_probe COMPILE_DEFINITIONS)
set(_expected_logging_policy
    "$<$<CONFIG:Debug>:HOLOSCAN_LOG_ACTIVE_LEVEL=HOLOSCAN_LOG_LEVEL_TRACE>"
    "$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:HOLOSCAN_LOG_ACTIVE_LEVEL=HOLOSCAN_LOG_LEVEL_DEBUG>"
    "$<$<CONFIG:MinSizeRel>:HOLOSCAN_LOG_ACTIVE_LEVEL=HOLOSCAN_LOG_LEVEL_INFO>"
)
if(NOT _compile_definitions STREQUAL _expected_logging_policy)
    message(FATAL_ERROR
        "Logging policy is not the expected target-private multi-config contract: "
        "${_compile_definitions}"
    )
endif()

include(BuildSecurityFlags)
get_target_property(_security_link_options holoscan_security_flags INTERFACE_LINK_OPTIONS)
get_target_property(_security_compile_options holoscan_security_flags INTERFACE_COMPILE_OPTIONS)
get_target_property(_security_compile_definitions
    holoscan_security_flags INTERFACE_COMPILE_DEFINITIONS)
if(_security_link_options OR _security_compile_options OR _security_compile_definitions)
    message(FATAL_ERROR "clang-cl probe received Linux/GNU security options")
endif()

# Exercise the positive classifier branch too. This proves the same helpers used by top-level
# sanitizer, RPATH, shared-link, security, and dependency-archive policy recognize the intended
# Linux GNU-like lane without needing to invoke a platform linker in this configure-only probe.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_CXX_COMPILER_ID GNU)
set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT GNU)
set(MSVC FALSE)
holoscan_platform_has_native_descriptor_readiness(_linux_has_descriptor_readiness)
if(NOT _linux_has_descriptor_readiness)
    message(FATAL_ERROR "Linux was classified as lacking native descriptor readiness")
endif()
holoscan_select_native_descriptor_readiness_source(
    _linux_readiness_source readiness_handle_linux.cpp readiness_handle_unsupported.cpp)
if(NOT _linux_readiness_source STREQUAL "readiness_handle_linux.cpp")
    message(FATAL_ERROR "Linux did not select the native readiness implementation")
endif()
holoscan_is_linux_gnu_like_frontend(_gnu_is_linux_gnu_like)
if(NOT _gnu_is_linux_gnu_like)
    message(FATAL_ERROR "GNU probe was not classified as a Linux GNU-like frontend")
endif()
holoscan_configure_linux_origin_rpath()
if(NOT CMAKE_BUILD_RPATH STREQUAL "\$ORIGIN:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" OR
   NOT CMAKE_INSTALL_RPATH STREQUAL "\$ORIGIN")
    message(FATAL_ERROR "GNU probe did not receive the expected ELF origin-relative RPATH")
endif()
add_library(gnu_archive_probe INTERFACE)
holoscan_configure_linux_archive_hiding(gnu_archive_probe libprobe)
get_target_property(_gnu_archive_link_options gnu_archive_probe INTERFACE_LINK_OPTIONS)
if(NOT _gnu_archive_link_options STREQUAL "LINKER:--exclude-libs,libprobe")
    message(FATAL_ERROR "GNU probe did not receive archive-hiding policy")
endif()

# Keep the active top-level call sites on the tested helpers. This static routing check complements
# the simulated target-property probe and prevents a future direct GNU option from bypassing it.
file(READ "${HOLOSCAN_SOURCE_DIR}/CMakeLists.txt" _top_level_cmake)
file(READ "${HOLOSCAN_SOURCE_DIR}/cmake/HoloMQ.cmake" _holomq_cmake)
file(READ "${HOLOSCAN_SOURCE_DIR}/cmake/modules/BuildSecurityFlags.cmake" _security_cmake)
file(READ "${HOLOSCAN_SOURCE_DIR}/cmake/deps/spdlog.cmake" _spdlog_cmake)
foreach(_required_top_level_call IN ITEMS
        "holoscan_is_linux_gnu_like_frontend(_holoscan_google_sanitizer_supported)"
        "holoscan_configure_linux_origin_rpath()")
    string(FIND "${_top_level_cmake}" "${_required_top_level_call}" _call_position)
    if(_call_position EQUAL -1)
        message(FATAL_ERROR "Top-level CMake bypassed tested policy: ${_required_top_level_call}")
    endif()
endforeach()
foreach(_readiness_cmake IN ITEMS "${_top_level_cmake}" "${_holomq_cmake}")
    string(FIND "${_readiness_cmake}"
        "holoscan_select_native_descriptor_readiness_source"
        _readiness_source_selection_position)
    if(_readiness_source_selection_position EQUAL -1)
        message(FATAL_ERROR "A readiness target bypassed the shared platform source selector")
    endif()
endforeach()
string(FIND "${_security_cmake}" "holoscan_is_linux_gnu_like_frontend" _security_call_position)
if(_security_call_position EQUAL -1)
    message(FATAL_ERROR "BuildSecurityFlags bypassed the tested frontend classifier")
endif()
foreach(_security_cxx_warning IN ITEMS
        "$<$<COMPILE_LANGUAGE:CXX>:-Wformat=2>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wformat-security>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
    string(FIND "${_security_cmake}" "${_security_cxx_warning}" _security_warning_position)
    if(_security_warning_position EQUAL -1)
        message(FATAL_ERROR
            "BuildSecurityFlags does not expose one CXX warning per generator expression: "
            "${_security_cxx_warning}"
        )
    endif()
endforeach()
string(FIND "${_security_cmake}"
    "$<$<COMPILE_LANGUAGE:CXX>:-Wformat=2 -Wformat-security -Werror>"
    _combined_security_warning_position)
if(NOT _combined_security_warning_position EQUAL -1)
    message(FATAL_ERROR "BuildSecurityFlags combines multiple CXX warnings in one list item")
endif()
string(FIND "${_spdlog_cmake}" "holoscan_configure_linux_archive_hiding(spdlog libspdlog)"
    _spdlog_call_position)
if(_spdlog_call_position EQUAL -1)
    message(FATAL_ERROR "spdlog archive hiding bypassed the tested helper")
endif()
