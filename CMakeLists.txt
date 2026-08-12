# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

set(HOLOSCAN_PACKAGE_NAME holoscan)

# ##############################################################################
# # Options
# ##############################################################################
option(BUILD_SHARED_LIBS "Build Shared Libraries" ON)
option(HOLOSCAN_BUILD_EXAMPLES "Build Holoscan SDK Examples" ON)
option(HOLOSCAN_BUILD_PYTHON "Build Holoscan SDK Python Bindings" ON)
option(HOLOSCAN_DOWNLOAD_DATASETS "Download SDK Datasets" ON)
option(HOLOSCAN_BUILD_TESTS "Build Holoscan SDK Tests" ON)
option(HOLOSCAN_BUILD_BENCHMARKS "Build Holoscan SDK Benchmarks" ON)
option(HOLOSCAN_USE_SCCACHE "Use sccache for building Holoscan SDK" ON)
option(HOLOSCAN_INSTALL_EXAMPLE_SOURCE "Install the example source code" ON)
option(HOLOSCAN_ENABLE_CLANG_TIDY "Enable use of clang-tidy" OFF)
option(HOLOSCAN_CLANG_TIDY_CHECK_HEADERS "Enable clang-tidy header checking (requires HOLOSCAN_ENABLE_CLANG_TIDY)" OFF)
set(HOLOSCAN_CLANG_TIDY_EXTRA_CHECKS "" CACHE STRING "Additional clang-tidy checks appended to the default check list")
option(HOLOSCAN_ENABLE_GOOGLE_SANITIZER "Enable use of google sanitizer" OFF)
option(HOLOSCAN_BUILD_CONFIG "Build the backend-neutral configuration component" ON)
option(HOLOSCAN_BUILD_CONFIG_IO "Build explicit-format configuration I/O" ON)
if(HOLOSCAN_BUILD_CONFIG_IO AND NOT HOLOSCAN_BUILD_CONFIG)
  message(FATAL_ERROR "HOLOSCAN_BUILD_CONFIG_IO requires HOLOSCAN_BUILD_CONFIG")
endif()
include(cmake/HoloscanToolingMetadata.cmake)
# Core always builds and links HoloMQ, but keep cache options so SDK builds can
# drive the embedded module's tests, examples, benchmarks, Python bindings, and
# default backend with the same switches HoloMQ uses when built standalone.
option(HOLOMQ_BUILD_TESTS "Build HoloMQ tests" ${HOLOSCAN_BUILD_TESTS})
option(HOLOMQ_BUILD_EXAMPLES "Build HoloMQ examples" ${HOLOSCAN_BUILD_EXAMPLES})
option(HOLOMQ_BUILD_BENCHMARKS "Build HoloMQ benchmarks" ${HOLOSCAN_BUILD_BENCHMARKS})
option(HOLOMQ_BUILD_PYTHON "Build HoloMQ Python bindings" ${HOLOSCAN_BUILD_PYTHON})
set(HOLOSCAN_MQ_BACKEND "iceoryx2" CACHE STRING "Default HoloMQ runtime backend for SDK integration")
set_property(CACHE HOLOSCAN_MQ_BACKEND PROPERTY STRINGS iceoryx2)
option(HOLOSCAN_ALLOW_SYSTEM_INSTALL "Allow installation to system directories" OFF)

# ##############################################################################
# # Prerequisite statements
# ##############################################################################

# Set the top-level directory of the project
set(HOLOSCAN_TOP "${CMAKE_CURRENT_LIST_DIR}")

# Append local cmake module path for CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH "${HOLOSCAN_TOP}/cmake/modules")

# Append local cmake dependency path CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${HOLOSCAN_TOP}/cmake/deps")

# Setup rapids-cmake
include(SetupRapidsCMake)
include(GenerateExportHeader)
include(HoloscanTargetConfiguration)

# ##############################################################################
# # Project definition
# ##############################################################################

# Set HOLOSCAN_BUILD_VERSION from 'VERSION' file
unset(HOLOSCAN_BUILD_VERSION CACHE)
file(STRINGS ${HOLOSCAN_TOP}/VERSION HOLOSCAN_BUILD_VERSION)

project(${HOLOSCAN_PACKAGE_NAME}
    VERSION ${HOLOSCAN_BUILD_VERSION}
    DESCRIPTION "Holoscan SDK"
    LANGUAGES C CXX
)
include(SetupCUDA) # CUDA Language enabled there after additional setup

# If enabling google sanitizer
if(HOLOSCAN_ENABLE_GOOGLE_SANITIZER)
  # Note: Before running the tests, the following command must be executed:
  #   export ASAN_OPTIONS=symbolize=1:protect_shadow_gap=0
  # This ensures that stack traces are symbolized and prevents false positives related to shadow gap
  # protection.
  # Without this, CUDA-related tests may fail (e.g., CUDA runtime API error "out of memory").
  # (See https://github.com/NVIDIA/DALI/pull/362).
  # You may still see some false positives when running the tests that uses the Vulkan-related APIs.
  holoscan_is_linux_gnu_like_frontend(_holoscan_google_sanitizer_supported)
  if(NOT _holoscan_google_sanitizer_supported)
    message(FATAL_ERROR
      "HOLOSCAN_ENABLE_GOOGLE_SANITIZER currently requires a Linux GNU-like C++ frontend"
    )
  endif()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-g>
    $<$<COMPILE_LANGUAGE:CXX>:-O1>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-stringop-truncation>
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>
    $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>
  )
  add_link_options($<$<LINK_LANGUAGE:CXX>:-fsanitize=address>)
  unset(_holoscan_google_sanitizer_supported)
endif()

# ##############################################################################
# # Global properties (CMAKE_*)
# ##############################################################################

# Setting the default paths for the build tree
# Libraries (shared and static) are put by default in the lib directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# `$ORIGIN` is an ELF loader convention. The common policy keeps it out of native MSVC, clang-cl,
# and other package metadata whose runtime-library placement follows a different contract.
holoscan_configure_linux_origin_rpath()

# Default component name for "cmake --install . --component <component>"
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "holoscan-dependencies")

# this generates a 'compile_commands.json' file which can be read by VSCode to
# configure the include paths for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Create a symbolic link for compile_commands.json in the source tree when building out of source
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(STATUS "Skipping compile_commands.json symlink because this is an in-source build.")
else()
    # Clangd searches parent directories and 'build/' folders for compile_commands.json.
    # To ensure Clangd can find it, we create a symlink in the source directory.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${CMAKE_BINARY_DIR}/compile_commands.json
                ${CMAKE_SOURCE_DIR}/compile_commands.json
        RESULT_VARIABLE result
        ERROR_VARIABLE error
    )
    if(result)
        message(FATAL_ERROR "Failed to create symbolic link for compile_commands.json: ${error}")
    endif()
endif()

# Avoid 'Up-to-date' install messages
set(CMAKE_INSTALL_MESSAGE LAZY)

# Default language settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Don't append suffix to shared libraries for debug build
set(CMAKE_DEBUG_POSTFIX "")

# ##############################################################################
# # RAPIDS CMAKE configurations
# # (https://docs.rapids.ai/api/rapids-cmake/stable/api.html)
# ##############################################################################

# Set default build type in case not specified
rapids_cmake_build_type(release)

# Generate a C++ header file (under ${CMAKE_CURRENT_BINARY_DIR}) that hold the version (X.Y.Z)
# information of the calling project
rapids_cmake_write_version_file(include/holoscan/version_config.hpp PREFIX HOLOSCAN)

# Get install lib dir
# (https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cmake_install_lib_dir.html)
rapids_cmake_install_lib_dir(HOLOSCAN_INSTALL_LIB_DIR)

# ##############################################################################
# # Setup Cache
# ##############################################################################

option(HOLOSCAN_USE_CACHE_DIR "Use CPM and compiler cache directory (set OFF for e.g. OpenEmbedded)" ON)
set(HOLOSCAN_CACHE_DIR ".cache" CACHE PATH "Directory for CPM and compiler caches (used when HOLOSCAN_USE_CACHE_DIR is ON).\
                                            If relative, it is relative to the source directory (CMAKE_SOURCE_DIR).")
mark_as_advanced(HOLOSCAN_CACHE_DIR)

# HOLOSCAN_USE_SCCACHE_SKIPPED will be set to TRUE by 'include(SetupCache)'
# if SCCache is not available.
set(HOLOSCAN_USE_SCCACHE_SKIPPED FALSE)
include(SetupCache)

# ##############################################################################
# # Setup hardening security build flags
# ##############################################################################
include(BuildSecurityFlags)

# ##############################################################################
# # Print project settings
# ##############################################################################
include(PrintProjectSettings)

# ##############################################################################
# # Define dependencies
# ##############################################################################

# Include pthread (add 'Threads::Threads' target to link against)
# (https://cmake.org/cmake/help/v3.18/module/FindThreads.html)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter QUIET)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/setup_dependencies.cmake")
message(STATUS "Configuring the minimal Holoscan SDK 5.x base")

if(HOLOSCAN_BUILD_TESTS OR HOLOSCAN_BUILD_BENCHMARKS)
    include(CTest)
endif()

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_test(NAME CMAKE_PORTABILITY.clang_cl_rejects_linux_gnu_options
        COMMAND ${CMAKE_COMMAND}
            -S "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/tests/target_configuration"
            -B "${CMAKE_CURRENT_BINARY_DIR}/cmake-tests/target-configuration-clang-cl"
            -DHOLOSCAN_MODULE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
            -DHOLOSCAN_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

if(Python3_Interpreter_FOUND)
    add_custom_target(check-export-macro-semantics
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_export_macro_semantics.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --compile-commands "${CMAKE_BINARY_DIR}/compile_commands.json"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Checking public-header export macro semantics"
        VERBATIM
    )
endif()

# If enabling clang-tidy
if(HOLOSCAN_ENABLE_CLANG_TIDY)
  find_program(_HOLOSCAN_CLANG_TIDY_EXECUTABLE NAMES clang-tidy NO_CACHE REQUIRED)
  execute_process(
    COMMAND "${_HOLOSCAN_CLANG_TIDY_EXECUTABLE}" --version
    OUTPUT_VARIABLE _HOLOSCAN_CLANG_TIDY_VERSION_OUTPUT
    ERROR_VARIABLE _HOLOSCAN_CLANG_TIDY_VERSION_ERROR
    RESULT_VARIABLE _HOLOSCAN_CLANG_TIDY_VERSION_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _HOLOSCAN_CLANG_TIDY_VERSION_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Failed to query clang-tidy version: ${_HOLOSCAN_CLANG_TIDY_VERSION_ERROR}")
  endif()
  # Common clang-tidy checks: enable all ("*"), then disable noisy or low-signal groups for this
  # codebase. Joined into the comma-separated form required by clang-tidy --checks=...
  # Wildcards (e.g. -cppcoreguidelines-*, -readability-*) subsume same-prefix checks; do not list
  # those checks again. Disable list is sorted alphabetically (after "*").
  set(_HOLOSCAN_CLANG_TIDY_CHECK_LIST
      "*"
      "-abseil-*"
      "-altera-*"
      "-android-*"
      "-bugprone-easily-swappable-parameters"
      "-bugprone-exception-escape"
      "-bugprone-implicit-widening-of-multiplication-result"
      "-bugprone-multi-level-implicit-pointer-conversion"
      "-bugprone-inc-dec-in-conditions"
      "-bugprone-casting-through-void"
      "-cert-*"
      "-clang-diagnostic-ignored-optimization-argument"
      "-clang-diagnostic-potentially-evaluated-expression"
      "-concurrency-mt-unsafe"
      "-cppcoreguidelines-*"
      "cppcoreguidelines-avoid-c-arrays"
      "cppcoreguidelines-init-variables"
      "cppcoreguidelines-missing-std-forward"
      "-fuchsia-*"
      "-google-*"
      "google-default-arguments"
      "-hicpp-*"
      "-llvm-else-after-return"
      # Keep disabled: HSDK 5 requires #pragma once, while this clang-tidy check
      # enforces the forbidden macro-guard convention.
      "-llvm-header-guard"
      "-llvm-include-order"
      "-llvm-qualified-auto"
      "-llvmlibc-callee-namespace"
      "-llvmlibc-implementation-in-namespace"
      "-llvmlibc-inline-function-decl"
      "-llvmlibc-restrict-system-libc-headers"
      "-misc-*"
      "misc-const-correctness"
      "misc-misplaced-const"
      "misc-unused-alias-decls"
      "-modernize-*"
      "modernize-pass-by-value"
      "-performance-enum-size"
      "-performance-inefficient-string-concatenation"
      "-performance-move-const-arg"
      "-performance-no-automatic-move"
      "-performance-no-int-to-ptr"
      "-portability-simd-intrinsics"
      "-readability-*"
      "readability-avoid-const-params-in-decls"
      "readability-else-after-return"
      "readability-function-cognitive-complexity"
      "readability-implicit-bool-conversion"
      "readability-inconsistent-declaration-parameter-name"
      "readability-redundant-inline-specifier")
  if(HOLOSCAN_CLANG_TIDY_EXTRA_CHECKS)
    list(APPEND _HOLOSCAN_CLANG_TIDY_CHECK_LIST "${HOLOSCAN_CLANG_TIDY_EXTRA_CHECKS}")
  endif()
  # LLVM 18 crashes in bugprone-argument-comment while parsing constrained Channel template calls
  # in public headers. Append the workaround last so extra checks cannot re-enable the crashing
  # checker; newer analyzer versions continue to exercise it.
  if(_HOLOSCAN_CLANG_TIDY_VERSION_OUTPUT MATCHES "version[ \\t]+18\\.")
    list(APPEND _HOLOSCAN_CLANG_TIDY_CHECK_LIST "-bugprone-argument-comment")
  endif()
  string(JOIN "," CLANG_TIDY_CHECKS ${_HOLOSCAN_CLANG_TIDY_CHECK_LIST})
  unset(_HOLOSCAN_CLANG_TIDY_CHECK_LIST)

  if(HOLOSCAN_CLANG_TIDY_CHECK_HEADERS)
    # Header filter to check project headers while excluding third-party code.
    # Note: --exclude-header-filter is not available until clang-tidy>=19.0, so we use
    # a positive regex that explicitly matches the directories we want to check.
    set(CLANG_TIDY_HEADER_FILTER ".*/include/holoscan/.*|.*/python/holoscan/.*")
    set(CMAKE_CXX_CLANG_TIDY
        "${_HOLOSCAN_CLANG_TIDY_EXECUTABLE}"
        "--header-filter=${CLANG_TIDY_HEADER_FILTER}"
        "--checks=${CLANG_TIDY_CHECKS}"
        "--extra-arg=-std=c++20"
    )
  else()
    # Without header filter, only check source files (not included headers)
    set(CMAKE_CXX_CLANG_TIDY
        "${_HOLOSCAN_CLANG_TIDY_EXECUTABLE}"
        "--checks=${CLANG_TIDY_CHECKS}"
        "--extra-arg=-std=c++20"
    )
  endif()
  unset(_HOLOSCAN_CLANG_TIDY_EXECUTABLE)
  unset(_HOLOSCAN_CLANG_TIDY_VERSION_ERROR)
  unset(_HOLOSCAN_CLANG_TIDY_VERSION_OUTPUT)
  unset(_HOLOSCAN_CLANG_TIDY_VERSION_RESULT)
endif()

# Suppress the warning: "warning #128: type qualifiers ignored on cast result type"
add_compile_options(
  $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=128>
)

# ##############################################################################
# # Package project
# ##############################################################################

# Establish the CPM / preset-package infrastructure before loading any dependency modules
# (rapids_cpm_find -> CPMAddPackage requires CPM to be initialized first).
rapids_cpm_init()

# HoloMQ dependencies
holoscan_add_dependency(iceoryx2)

# Core library dependencies.
holoscan_add_dependency(expected_rapids)
holoscan_add_dependency(fmt_rapids)
# magic_enum backs error_code_name() (enum -> name) in core/errors.hpp; header-only.
holoscan_add_dependency(magic_enum)
# DLPack needed for Tensor interop
holoscan_add_dependency(dlpack_rapids)
# CCCL provides <cuda/stream> (cuda::stream_ref) used by the Tensor stream-aware API.
holoscan_add_dependency(cccl COMPONENT holoscan-dependencies)
# spdlog backs the holoscan logger; built static with header-only fmt (see cmake/deps/spdlog.cmake).
holoscan_add_dependency(spdlog)
# FlatBuffers provides flatc (schema codegen) + the runtime lib for schema-backed Tensor
# materialization (see cmake/deps/flatbuffers.cmake). Both targets are EXCLUDE_FROM_ALL; they build
# only when a schema/codegen target depends on them.
holoscan_add_dependency(flatbuffers)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/HoloscanSchemaTools.cmake")

# Optional cold configuration tooling remains independent of holoscan::core and CUDA-facing
# targets. The config target owns only Standard C++ value/codec/digest behavior; filesystem and
# parser code is isolated in config_io.
if(HOLOSCAN_BUILD_CONFIG)
  add_library(config
      src/config/document.cpp
  )
  add_library(${HOLOSCAN_PACKAGE_NAME}::config ALIAS config)
  holoscan_configure_cxx20_target(config PUBLIC)
  target_include_directories(config
      PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR}/src
      PUBLIC
          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
          $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
          $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(config PUBLIC tl::expected)
  # REQ: CONFIG-015. Keep this lightweight target independent from Core, CUDA, parsers, and I/O.
  get_target_property(_holoscan_config_link_libraries config LINK_LIBRARIES)
  foreach(_holoscan_config_dependency IN LISTS _holoscan_config_link_libraries)
    string(TOLOWER "${_holoscan_config_dependency}" _holoscan_config_dependency_lower)
    if(_holoscan_config_dependency_lower MATCHES "core|cuda|config_io")
      message(FATAL_ERROR
        "holoscan::config gained a forbidden dependency: ${_holoscan_config_dependency}"
      )
    endif()
  endforeach()
  unset(_holoscan_config_dependency_lower)
  unset(_holoscan_config_dependency)
  unset(_holoscan_config_link_libraries)
  target_compile_definitions(config
      PUBLIC $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:HOLOSCAN_CONFIG_STATIC_DEFINE>
  )
  generate_export_header(config
      BASE_NAME HOLOSCAN_CONFIG
      EXPORT_MACRO_NAME HOLOSCAN_CONFIG_EXPORT
      NO_EXPORT_MACRO_NAME HOLOSCAN_CONFIG_NO_EXPORT
      DEPRECATED_MACRO_NAME HOLOSCAN_CONFIG_DEPRECATED
      STATIC_DEFINE HOLOSCAN_CONFIG_STATIC_DEFINE
      INCLUDE_GUARD_NAME HOLOSCAN_CONFIG_EXPORT_H_GENERATED
      EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_config_export.h"
  )
  set_target_properties(config PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      EXPORT_NAME config
      OUTPUT_NAME holoscan_config
      POSITION_INDEPENDENT_CODE ON
      SOVERSION ${PROJECT_VERSION_MAJOR}
      VERSION ${PROJECT_VERSION}
      VISIBILITY_INLINES_HIDDEN YES
  )
  holoscan_configure_linux_shared_link(config)
  list(APPEND HOLOSCAN_CONFIG_INSTALL_TARGETS config)
  holoscan_install_tooling_metadata(config.json.in holoscan-config.json holoscan-config)

  if(HOLOSCAN_BUILD_CONFIG_IO)
    # Parser backends are private replaceable implementation details. yyjson validates strict JSON
    # syntax, while Holoscan retains semantic conversion, limits, errors, canonical bytes, and the
    # persistence state machine; no backend type or package target crosses the public interface.
    holoscan_add_dependency(yaml-cpp_rapids COMPONENT holoscan-config-io)
    holoscan_add_dependency(yyjson)
    if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
      # reflect-cpp is differential test evidence for the MR !4537 typed-backend direction. It is
      # neither a production parse pass nor a production package/build dependency.
      holoscan_add_dependency(reflectcpp)
    endif()
    get_target_property(_holoscan_config_yaml_target_type yaml-cpp::yaml-cpp TYPE)
    if(NOT _holoscan_config_yaml_target_type STREQUAL "STATIC_LIBRARY")
      message(FATAL_ERROR
          "The private config_io parser provider must be a static yaml-cpp archive; got "
          "${_holoscan_config_yaml_target_type}"
      )
    endif()
    unset(_holoscan_config_yaml_target_type)
    # Build yyjson once as private position-independent objects. The production config_io library
    # absorbs these objects, while the reflect-cpp differential test adapter reuses them without
    # turning reflect-cpp into a production dependency or exporting a yyjson package contract.
    add_library(holoscan_config_yyjson_private_objects OBJECT
        "${yyjson_SOURCE_DIR}/src/yyjson.c"
    )
    target_include_directories(holoscan_config_yyjson_private_objects
        SYSTEM PRIVATE "${yyjson_SOURCE_DIR}/src"
    )
    # yyjson intentionally marks its C API with default ELF visibility. Override that configurable
    # declaration macro so the embedded provider obeys the owning target's hidden-symbol policy.
    target_compile_definitions(holoscan_config_yyjson_private_objects PRIVATE yyjson_api=)
    set_target_properties(holoscan_config_yyjson_private_objects PROPERTIES
        C_CLANG_TIDY ""
        C_EXTENSIONS NO
        C_STANDARD 11
        C_STANDARD_REQUIRED YES
        C_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
    )
    add_library(config_io
        src/config/io.cpp
        src/config/json_yyjson.cpp
        $<TARGET_OBJECTS:holoscan_config_yyjson_private_objects>
    )
    if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
      add_library(holoscan_config_reflect_private_objects OBJECT
          "${reflectcpp_SOURCE_DIR}/src/reflectcpp.cpp"
          "${reflectcpp_SOURCE_DIR}/src/reflectcpp_json.cpp"
      )
      holoscan_configure_cxx20_target(holoscan_config_reflect_private_objects PRIVATE)
      target_compile_definitions(holoscan_config_reflect_private_objects PRIVATE yyjson_api=)
      target_include_directories(holoscan_config_reflect_private_objects
          SYSTEM PRIVATE
              "${reflectcpp_SOURCE_DIR}/include"
              "${reflectcpp_SOURCE_DIR}/include/rfl/thirdparty"
      )
      set_target_properties(holoscan_config_reflect_private_objects PROPERTIES
          CXX_CLANG_TIDY ""
          POSITION_INDEPENDENT_CODE ON
      )
    endif()
    add_library(${HOLOSCAN_PACKAGE_NAME}::config_io ALIAS config_io)
    holoscan_configure_cxx20_target(config_io PUBLIC)
    target_include_directories(config_io
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_include_directories(config_io SYSTEM PRIVATE
        ${yyjson_SOURCE_DIR}/src
    )
    target_link_libraries(config_io
        PUBLIC holoscan::config
        PRIVATE
            $<BUILD_INTERFACE:yaml-cpp::yaml-cpp>
            $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:
                $<INSTALL_INTERFACE:holoscan::_config_yaml_private>>
    )
    target_compile_definitions(config_io
        PRIVATE
            HOLOSCAN_CONFIG_IO_USE_YAML_CPP=1
            yyjson_api=
    )
    target_compile_definitions(config_io
        PUBLIC $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:HOLOSCAN_CONFIG_IO_STATIC_DEFINE>
    )
    generate_export_header(config_io
        BASE_NAME HOLOSCAN_CONFIG_IO
        EXPORT_MACRO_NAME HOLOSCAN_CONFIG_IO_EXPORT
        NO_EXPORT_MACRO_NAME HOLOSCAN_CONFIG_IO_NO_EXPORT
        DEPRECATED_MACRO_NAME HOLOSCAN_CONFIG_IO_DEPRECATED
        STATIC_DEFINE HOLOSCAN_CONFIG_IO_STATIC_DEFINE
        INCLUDE_GUARD_NAME HOLOSCAN_CONFIG_IO_EXPORT_H_GENERATED
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_config_io_export.h"
    )
    set_target_properties(config_io PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        EXPORT_NAME config_io
        OUTPUT_NAME holoscan_config_io
        POSITION_INDEPENDENT_CODE ON
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION}
        VISIBILITY_INLINES_HIDDEN YES
    )
    holoscan_configure_linux_shared_link(config_io)
    list(APPEND HOLOSCAN_CONFIG_IO_INSTALL_TARGETS config_io)
    holoscan_install_tooling_metadata(
        config_io.json.in holoscan-config-io.json holoscan-config-io)
  endif()
endif()

# Standalone logger library. Built as its own DSO (libholoscan_logger.so), independent of
# holoscan core, so that other modules (e.g. under public/modules/) can depend on logging without
# pulling in any other core functionality.
add_library(logger
    src/logger/logger.cpp
    src/logger/spdlog_logger.cpp
)
add_library(${HOLOSCAN_PACKAGE_NAME}::logger ALIAS logger)
holoscan_configure_cxx20_target(logger PUBLIC)
holoscan_apply_logging_policy(logger)
target_include_directories(logger
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
# logger.hpp exposes fmt in its public API; spdlog is a header-only implementation detail of the
# backend (matching 4.x). Header-only spdlog is compiled directly into the PIC shared logger
# object, so no separate static libspdlog.a is linked (which would otherwise require -fPIC).
target_link_libraries(logger PUBLIC fmt::fmt-header-only PRIVATE spdlog::spdlog_header_only)
target_compile_definitions(logger
    PUBLIC $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:HOLOSCAN_LOGGER_STATIC_DEFINE>
)
generate_export_header(logger
    BASE_NAME HOLOSCAN_LOGGER
    EXPORT_MACRO_NAME HOLOSCAN_LOGGER_EXPORT
    NO_EXPORT_MACRO_NAME HOLOSCAN_LOGGER_NO_EXPORT
    DEPRECATED_MACRO_NAME HOLOSCAN_LOGGER_DEPRECATED
    STATIC_DEFINE HOLOSCAN_LOGGER_STATIC_DEFINE
    INCLUDE_GUARD_NAME HOLOSCAN_LOGGER_EXPORT_H_GENERATED
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_logger_export.h"
)
set_target_properties(logger PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    EXPORT_NAME logger
    OUTPUT_NAME holoscan_logger
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    VISIBILITY_INLINES_HIDDEN YES
)
holoscan_configure_linux_shared_link(logger)
list(APPEND HOLOSCAN_INSTALL_TARGETS logger)

# Minimal SDK target. The Holoscan Runtime implementation will replace this
# skeleton library as the 5.x SDK code lands.
add_library(core
    src/holoscan.cpp
    src/channel/channel.cpp
    src/channel/channel_catalog_binding.cpp
    src/channel/channel_common.cpp
    src/channel/channel_descriptor.cpp
    src/channel/channel_inline_payload.cpp
    src/channel/channel_tensor.cpp
    src/channel/channel_tensor_loan.cpp
    src/channel/channel_tensor_schema_payload.cpp
    src/channel/catalog_tensor_receive_control.cpp
    src/channel/endpoint_catalog.cpp
    src/channel/endpoint_catalog_builder.cpp
    src/channel/holomq_tensor.cpp
    src/channel/pod_codec.cpp
    src/channel/process_identity.cpp
    src/compiler/compile.cpp
    src/compiler/artifact_emitter.cpp
    src/compiler/partition_plan_lowering.cpp
    src/core/errors.cpp
    src/core/execution_context.cpp
    src/core/execution_plan.cpp
    src/core/graph.cpp
    src/core/lifecycle.cpp
    src/core/operator.cpp
    src/core/operator_spec.cpp
    src/core/parameter.cpp
    src/core/parameter_catalog.cpp
    src/core/temporal_contract.cpp
    src/core/temporal_contract_lowering.cpp
    src/core/temporal_contract_records.cpp
    src/core/tensor_output_loan.cpp
    src/core/tensor_port_type_ops.cpp
    src/core/domain/tensor.cpp
    src/core/domain/dlpack_utils.cpp
    src/cuda_execution/cuda_event_pool.cpp
    src/cuda_execution/cuda_green_context.cpp
    src/cuda_execution/cuda_green_context_pool.cpp
    src/cuda_execution/cuda_stream_pool.cpp
    src/memory/block_memory_pool.cpp
    src/memory/cuda_memory_resource_allocator.cpp
    src/memory/tensor_storage.cpp
    src/memory/unbounded_allocator.cpp
    src/runtime/executor.cpp
    src/runtime/dispatcher.cpp
    src/runtime/endpoint_clock_health.cpp
    src/runtime/endpoint_clock_health_runtime.cpp
    src/runtime/lifecycle_control_state.cpp
    src/runtime/lifecycle_executor.cpp
    src/runtime/physical_readiness_binding.cpp
    src/runtime/runtime_notification_source.cpp
    src/runtime/runtime_callback_context.cpp
    src/runtime/runtime_channel_binding.cpp
    src/runtime/runtime_materialization.cpp
    src/runtime/runtime_tensor_resources.cpp
    src/runtime/parameter_catalog_projection.cpp
    src/runtime/parameter_control_state.cpp
    src/runtime/parameter_update_coordinator.cpp
    src/runtime/run_session.cpp
    src/runtime/scheduling/deterministic_serial_ready_policy.cpp
    src/runtime/scheduling/ready_policy_catalog.cpp
    src/runtime/execution_lane.cpp
    src/threading/thread_placement.cpp
    src/time/clock_registry.cpp
    src/time/manual_clock.cpp
    src/time/realtime_clock.cpp
    src/time/synthetic_clock.cpp
)
add_library(${HOLOSCAN_PACKAGE_NAME}::core ALIAS core)
holoscan_configure_cxx20_target(core PUBLIC)
holoscan_apply_logging_policy(core)
target_include_directories(core
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/holomq/include
        ${CMAKE_CURRENT_BINARY_DIR}/modules/holomq/include
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(core
    PUBLIC
        CCCL::CCCL
        CUDA::cuda_driver
        CUDA::cudart
        dlpack::dlpack
        magic_enum::magic_enum
        tl::expected
        # memory/memory_buffer.hpp exposes fmt in its public API, so fmt is a public compile
        # dependency of core (matching 4.x and the logger DSO). fmt-header-only is an INTERFACE
        # target, so it does not need to be in the install export set.
        fmt::fmt-header-only
        # memory/memory_buffer.hpp logs via HOLOSCAN_LOG_* inline in a public header, so
        # holoscan::logger (which provides Logger::log_message) is a public dependency, matching
        # 4.x where logging is part of the public API. It is a holoscan export-set target.
        holoscan::logger
    PRIVATE
        Threads::Threads
)
target_compile_definitions(core
    PUBLIC $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:HOLOSCAN_CORE_STATIC_DEFINE>
)
generate_export_header(core
    BASE_NAME HOLOSCAN_CORE
    EXPORT_MACRO_NAME HOLOSCAN_CORE_EXPORT
    NO_EXPORT_MACRO_NAME HOLOSCAN_CORE_NO_EXPORT
    DEPRECATED_MACRO_NAME HOLOSCAN_CORE_DEPRECATED
    STATIC_DEFINE HOLOSCAN_CORE_STATIC_DEFINE
    INCLUDE_GUARD_NAME HOLOSCAN_CORE_EXPORT_H_GENERATED
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_core_export.h"
)
set_target_properties(core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    EXPORT_NAME core
    OUTPUT_NAME holoscan_core
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    VISIBILITY_INLINES_HIDDEN YES
)
holoscan_configure_linux_shared_link(core)
list(APPEND HOLOSCAN_INSTALL_TARGETS core)

add_subdirectory(schemas)
list(APPEND HOLOSCAN_INSTALL_TARGETS holoscan_schema)

if(NOT DEFINED HOLOMQ_DEFAULT_BACKEND)
    set(HOLOMQ_DEFAULT_BACKEND "${HOLOSCAN_MQ_BACKEND}" CACHE STRING "Default HoloMQ runtime backend")
    set_property(CACHE HOLOMQ_DEFAULT_BACKEND PROPERTY STRINGS iceoryx2)
endif()
# RAPIDS CPM must be initialized (via SetupRapidsCMake above) before this
# subdirectory is processed. HoloMQ dependency helpers call rapids_cpm_find().
add_subdirectory(modules/holomq)
include(${rapids-cmake-dir}/export/package.cmake)
if(NOT BUILD_SHARED_LIBS)
    # Static archives retain private libraries as link-only usage requirements. Register the
    # remaining separately installed packages before exposing static SDK targets downstream.
    rapids_export_package(INSTALL holomq ${HOLOSCAN_PACKAGE_NAME}-exports)
    rapids_export_package(INSTALL spdlog ${HOLOSCAN_PACKAGE_NAME}-exports)
    rapids_export_package(INSTALL Flatbuffers ${HOLOSCAN_PACKAGE_NAME}-exports)
endif()
# Core uses HoloMQ for Channel descriptor transport, fixed-size inline payload buckets, and
# cross-process opaque buffer references. HoloMQ remains a source-private implementation detail;
# static exports retain it only as a link-only usage requirement.
target_link_libraries(core PRIVATE holomq::holomq)
target_compile_definitions(core
    PRIVATE
        HOLOMQ_TRANSPORT_IOX2=1
)

# The compiler conformance tests exercise private lowering seams without exporting
# those seams from holoscan::core. Compile the shared implementation once and add
# the resulting objects to both core and compile_test; this avoids independently
# compiling the same translation units with potentially divergent flags.
add_library(holoscan_compiler_private_objects OBJECT
    src/compiler/closed_island_activation_plan.cpp
    src/compiler/compile_platform.cpp
    src/compiler/deployment_plan_lowering.cpp
    src/compiler/inline_partition_execution_plan.cpp
    src/compiler/partition_runner_plan.cpp
    src/compiler/plan_artifact_lowering.cpp
    src/compiler/parameter_plan_lowering.cpp
    src/compiler/physical_plan_lowering.cpp
    src/compiler/process_local_channel_plan_validation.cpp
    src/compiler/process_plan_lowering.cpp
    src/compiler/profile_admission.cpp
    src/compiler/profile_manifest_lowering.cpp
    src/compiler/runtime_package_capabilities.cpp
    src/runtime/runtime_package_capability.cpp
    src/compiler/temporal_plan_lowering.cpp
    src/threading/linux_thread_controller.cpp
    src/compiler/partitioning/partition_plan_validator.cpp
    src/compiler/partitioning/partition_planner_catalog.cpp
    src/compiler/partitioning/partitioning_pass.cpp
    src/compiler/partitioning/serial_baseline_partition_planner.cpp
    src/compiler/partitioning/source_chain_partition_planner.cpp
)
holoscan_configure_cxx20_target(holoscan_compiler_private_objects PRIVATE)
holoscan_apply_logging_policy(holoscan_compiler_private_objects)
target_include_directories(holoscan_compiler_private_objects
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/holomq/include
        ${CMAKE_CURRENT_BINARY_DIR}/modules/holomq/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
)
target_link_libraries(holoscan_compiler_private_objects
    PRIVATE
        CCCL::CCCL
        CUDA::cuda_driver
        CUDA::cudart
        dlpack::dlpack
        magic_enum::magic_enum
        tl::expected
        fmt::fmt-header-only
        holoscan::logger
        holomq::holomq
)
target_compile_definitions(holoscan_compiler_private_objects
    PRIVATE
        core_EXPORTS
        HOLOMQ_TRANSPORT_IOX2=1
        $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:HOLOSCAN_CORE_STATIC_DEFINE>
)
set_target_properties(holoscan_compiler_private_objects PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN YES
)

# Lifecycle plan lowering and its pure normalization kernels are likewise
# private implementation shared by core and focused conformance targets.
add_library(holoscan_lifecycle_plan_private_objects OBJECT
    src/compiler/lifecycle_plan_lowering.cpp
    src/core/lifecycle_graph_lowering.cpp
    src/core/lifecycle_kernel.cpp
    src/core/lifecycle_transaction_kernel.cpp
)
holoscan_configure_cxx20_target(holoscan_lifecycle_plan_private_objects PRIVATE)
holoscan_apply_logging_policy(holoscan_lifecycle_plan_private_objects)
target_include_directories(holoscan_lifecycle_plan_private_objects
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/modules/holomq/include
        ${CMAKE_CURRENT_BINARY_DIR}/modules/holomq/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
)
target_link_libraries(holoscan_lifecycle_plan_private_objects
    PRIVATE
        CCCL::CCCL
        CUDA::cuda_driver
        CUDA::cudart
        dlpack::dlpack
        magic_enum::magic_enum
        tl::expected
        fmt::fmt-header-only
        holoscan::logger
        holomq::holomq
)
target_compile_definitions(holoscan_lifecycle_plan_private_objects
    PRIVATE
        core_EXPORTS
        HOLOMQ_TRANSPORT_IOX2=1
        $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:HOLOSCAN_CORE_STATIC_DEFINE>
)
set_target_properties(holoscan_lifecycle_plan_private_objects PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN YES
)
target_sources(core PRIVATE
    $<TARGET_OBJECTS:holoscan_compiler_private_objects>
    $<TARGET_OBJECTS:holoscan_lifecycle_plan_private_objects>
)

include(cmake/HoloscanParameterTools.cmake)

if(HOLOSCAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(HOLOSCAN_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

# A clean prefix receives only the configured Core library type. If this prefix was used by a
# previous build with the opposite BUILD_SHARED_LIBS value, remove that stale Core artifact.
if(BUILD_SHARED_LIBS)
    install(CODE
        "file(REMOVE \"\${CMAKE_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}holoscan_core${CMAKE_STATIC_LIBRARY_SUFFIX}\")"
    )
else()
    install(CODE
        "file(GLOB _holoscan_core_shared_libraries
            \"\${CMAKE_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}holoscan_core${CMAKE_SHARED_LIBRARY_SUFFIX}*\"
        )
        file(REMOVE \${_holoscan_core_shared_libraries})"
    )
endif()

install(TARGETS ${HOLOSCAN_INSTALL_TARGETS}
    DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}
    EXPORT ${HOLOSCAN_PACKAGE_NAME}-exports
    COMPONENT holoscan-core
)

if(HOLOSCAN_BUILD_CONFIG)
  install(TARGETS ${HOLOSCAN_CONFIG_INSTALL_TARGETS}
      DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}
      EXPORT ${HOLOSCAN_PACKAGE_NAME}-config-exports
      COMPONENT holoscan-config
  )
  install(EXPORT ${HOLOSCAN_PACKAGE_NAME}-config-exports
      FILE ${HOLOSCAN_PACKAGE_NAME}-config-targets.cmake
      NAMESPACE ${HOLOSCAN_PACKAGE_NAME}::
      DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}/cmake/${HOLOSCAN_PACKAGE_NAME}
      COMPONENT holoscan-config
  )
  # The lightweight component is independently installable and therefore carries its sole public
  # checked-result dependency instead of requiring the aggregate dependency component.
  install(DIRECTORY "${tl-expected_SOURCE_DIR}/include/tl"
      DESTINATION include
      COMPONENT holoscan-config
  )
  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/configs/tl-expected-config.cmake"
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/configs/tl-expected-config-version.cmake"
      DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/tl-expected"
      COMPONENT holoscan-config
  )
endif()

if(HOLOSCAN_BUILD_CONFIG_IO)
  install(TARGETS ${HOLOSCAN_CONFIG_IO_INSTALL_TARGETS}
      DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}
      EXPORT ${HOLOSCAN_PACKAGE_NAME}-config-io-exports
      COMPONENT holoscan-config-io
  )
  install(EXPORT ${HOLOSCAN_PACKAGE_NAME}-config-io-exports
      FILE ${HOLOSCAN_PACKAGE_NAME}-config-io-targets.cmake
      NAMESPACE ${HOLOSCAN_PACKAGE_NAME}::
      DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}/cmake/${HOLOSCAN_PACKAGE_NAME}
      COMPONENT holoscan-config-io
  )
  if(NOT BUILD_SHARED_LIBS)
    # Static consumers receive the qualified parser archive as an SDK-private provider. The
    # exported config_io target refers only to this Holoscan-owned imported target; yaml-cpp
    # remains absent from downstream package discovery and public target names.
    install(FILES $<TARGET_FILE:yaml-cpp::yaml-cpp>
        DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}
        RENAME
            ${CMAKE_STATIC_LIBRARY_PREFIX}holoscan_config_yaml_private${CMAKE_STATIC_LIBRARY_SUFFIX}
        COMPONENT holoscan-config-io
    )
  endif()
endif()

# Copy version file
install(FILES ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/include/holoscan/version_config.hpp
    DESTINATION include/holoscan
    COMPONENT holoscan-core
)

install(FILES ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/include/holoscan/holoscan_core_export.h
    DESTINATION include/holoscan
    COMPONENT holoscan-core
)

install(FILES ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/include/holoscan/holoscan_logger_export.h
    DESTINATION include/holoscan
    COMPONENT holoscan-core
)

if(HOLOSCAN_BUILD_CONFIG)
  install(FILES ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/include/holoscan/holoscan_config_export.h
      DESTINATION include/holoscan
      COMPONENT holoscan-config
  )
  install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/config/document.hpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/config/codec.hpp"
      DESTINATION include/holoscan/config
      COMPONENT holoscan-config
  )
endif()

if(HOLOSCAN_BUILD_CONFIG_IO)
  install(FILES ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/include/holoscan/holoscan_config_io_export.h
      DESTINATION include/holoscan
      COMPONENT holoscan-config-io
  )
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/config/io.hpp"
      DESTINATION include/holoscan/config
      COMPONENT holoscan-config-io
  )
endif()

# Public logger header (the holoscan::logger / HOLOSCAN_LOG_* logging API). Installed so consumers can
# log against the installed SDK. spdlog_logger.hpp is the spdlog-backed implementation detail and is
# intentionally not part of the public install surface.
install(FILES "${CMAKE_SOURCE_DIR}/include/holoscan/logger/logger.hpp"
    DESTINATION include/holoscan/logger
    COMPONENT holoscan-core
)

install(FILES "${CMAKE_SOURCE_DIR}/include/holoscan/holoscan.hpp"
    DESTINATION include/holoscan
    COMPONENT holoscan-core
)

# Public channel headers. Installed so consumers can use typed Channel APIs and schema payload
# wrappers from the installed SDK.
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/holoscan/channel"
    DESTINATION include/holoscan
    COMPONENT holoscan-core
    FILES_MATCHING PATTERN "*.hpp"
)

# Public core headers (the expected<T, Error> error-handling API). Installed so consumers can use
# holoscan::expected / Error / error_code_name against the installed SDK.
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/holoscan/core"
    DESTINATION include/holoscan
    COMPONENT holoscan-core
    FILES_MATCHING PATTERN "*.hpp"
)

# Public CUDA execution headers. Installed so consumers can use stream, event, and green-context
# pool APIs against the installed SDK.
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/holoscan/cuda_execution"
    DESTINATION include/holoscan
    COMPONENT holoscan-core
    FILES_MATCHING PATTERN "*.hpp"
)

# Public memory headers. Installed so consumers can use allocator and tensor storage APIs against
# the installed SDK.
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/holoscan/memory"
    DESTINATION include/holoscan
    COMPONENT holoscan-core
    FILES_MATCHING PATTERN "*.hpp"
)

# Public threading headers (the ThreadPlacement value API). Installed so consumers can configure
# thread affinity and scheduling against the installed SDK.
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/holoscan/threading"
    DESTINATION include/holoscan
    COMPONENT holoscan-core
    FILES_MATCHING PATTERN "*.hpp"
)

# Public clock headers. Installed so consumers can use clock APIs against the installed SDK.
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/holoscan/time"
    DESTINATION include/holoscan
    COMPONENT holoscan-core
    FILES_MATCHING PATTERN "*.hpp"
)

# Apache 2.0 license at install prefix root
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE.txt"
    DESTINATION .
    COMPONENT holoscan-core
)

if(HOLOSCAN_BUILD_PYTHON)
    add_subdirectory(python/holoscan)
endif()

if(HOLOSCAN_BUILD_PYTHON AND HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(python/tests)
endif()

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Install CMake scripts to download data from NGC so it can be reused by HoloHub
install(FILES "${CMAKE_SOURCE_DIR}/cmake/modules/HoloscanDownloadData.cmake"
DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
COMPONENT "holoscan-core"
)
# FindTinyXML2.cmake: used by holoscan-config.cmake when TinyXML2Config.cmake is absent
# (e.g. Debian libtinyxml2-dev). Must live next to holoscan-config.cmake on CMAKE_MODULE_PATH.
install(FILES "${CMAKE_SOURCE_DIR}/cmake/modules/FindTinyXML2.cmake"
        DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
        COMPONENT "holoscan-core"
)
# Shipped examples are independent package consumers but retain the SDK's build-configuration
# logging matrix. Install only that narrow caller policy, not the complete in-tree target policy.
install(FILES "${CMAKE_SOURCE_DIR}/cmake/modules/HoloscanLoggingPolicy.cmake"
        DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
        COMPONENT "holoscan-core"
)
install(FILES "${CMAKE_SOURCE_DIR}/cmake/HoloscanGraphArtifacts.cmake"
        DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
        COMPONENT "holoscan"
)
install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HoloscanSchemaTools.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HoloscanGenerateSchemaTraits.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HoloscanGenerateDescriptorSchemaTraits.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HoloscanSchemaTraitsConfig.cmake.in"
        DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
        COMPONENT "holoscan"
)

# Define docs and hooks
set(holoscan_doc_string [=[
libholoscan: Holoscan SDK C++ API
]=])

# Defines the install export hook.
set(holoscan_install_hook_code_string [=[
include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit)
find_dependency(Threads)

# CUDA ships the CCCL (and CUB/Thrust/libcudacxx) CMake package configs in a
# layout-dependent location: the per-target directory (<target>/lib/cmake/cccl),
# the toolkit root lib dir (<root>/lib64/cmake/cccl), or the detected library
# dir. CMake does not auto-search lib64 when CMAKE_LIBRARY_ARCHITECTURE is set,
# so prepend the cmake directory that actually contains cccl/ directly. This lets
# installed consumers (and MatX's find_dependency(CCCL)) resolve CCCLConfig.cmake
# regardless of the CUDA toolkit on-disk layout.
foreach(_holoscan_cccl_cmake_dir
    "${CUDAToolkit_TARGET_DIR}/lib/cmake"
    "${CUDAToolkit_TARGET_DIR}/lib64/cmake"
    "${CUDAToolkit_LIBRARY_DIR}/cmake"
    "${CUDAToolkit_LIBRARY_ROOT}/lib64/cmake"
    "${CUDAToolkit_LIBRARY_ROOT}/lib/cmake"
    "/usr/local/cuda/lib64/cmake")
  if(IS_DIRECTORY "${_holoscan_cccl_cmake_dir}/cccl"
     AND NOT _holoscan_cccl_cmake_dir IN_LIST CMAKE_PREFIX_PATH)
    list(PREPEND CMAKE_PREFIX_PATH "${_holoscan_cccl_cmake_dir}")
  endif()
endforeach()
unset(_holoscan_cccl_cmake_dir)

get_filename_component(_holoscan_install_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT _holoscan_install_prefix IN_LIST CMAKE_PREFIX_PATH)
  list(PREPEND CMAKE_PREFIX_PATH "${_holoscan_install_prefix}")
endif()
unset(_holoscan_install_prefix)
]=])
string(CONFIGURE "${holoscan_install_hook_code_string}" holoscan_install_hook_code_string @ONLY)

set(holoscan_build_hook_code_string [=[
]=])


# The generated holoscan-dependencies.cmake runs before the final holoscan-config.cmake hook.
# Register the SDK install prefix for packages shipped alongside Holoscan so RAPIDS emits their
# package roots before its generated find_dependency() calls.
include(${rapids-cmake-dir}/export/find_package_root.cmake)
set(_holoscan_install_package_roots TRUE)
# FlatBuffers headers occur in Holoscan's public schema headers. The bundled package supplies the
# matching runtime target to both shared and static SDK consumers.
set(_holoscan_install_packages
    tl-expected
    fmt
    magic_enum
    dlpack
    CCCL
    Flatbuffers
)
if(NOT BUILD_SHARED_LIBS)
    # Static archives propagate private libraries as link-only requirements. Shared-package
    # consumers must not need a public HoloMQ or iceoryx2 package target.
    list(APPEND _holoscan_install_packages
        holomq
        iceoryx2-cxx
        iceoryx2-bb-cxx
        iceoryx2-c
        spdlog
    )
endif()
foreach(_holoscan_install_package IN LISTS _holoscan_install_packages)
    rapids_export_find_package_root(INSTALL
        "${_holoscan_install_package}"
        "\${CMAKE_CURRENT_LIST_DIR}/../../.."
        EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
        CONDITION _holoscan_install_package_roots
    )
endforeach()
unset(_holoscan_install_package)
unset(_holoscan_install_packages)
unset(_holoscan_install_package_roots)

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_export.html#rapids-export
# (rapids_export() always installs to lib and doesn't use GNUInstallDirs)
rapids_export(
    INSTALL ${HOLOSCAN_PACKAGE_NAME}
    EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    NAMESPACE ${HOLOSCAN_PACKAGE_NAME}::
    VERSION OFF # Forces to not generate the config version file
    DOCUMENTATION holoscan_doc_string
    FINAL_CODE_BLOCK holoscan_install_hook_code_string
)

# RAPIDS generates the complete Core package configuration. Preserve that implementation behind a
# component-aware front door so a config-only consumer never executes Core's CUDA/CCCL/Threads
# discovery or imports Core targets. The lightweight component exports above contain only the
# config targets and their public checked-result dependency.
include(CMakePackageConfigHelpers)
set(_holoscan_rapids_package_dir
    "${PROJECT_BINARY_DIR}/rapids-cmake/${HOLOSCAN_PACKAGE_NAME}/export/${HOLOSCAN_PACKAGE_NAME}"
)
set(_holoscan_rapids_config
    "${_holoscan_rapids_package_dir}/${HOLOSCAN_PACKAGE_NAME}-config.cmake"
)
set(_holoscan_full_config
    "${_holoscan_rapids_package_dir}/${HOLOSCAN_PACKAGE_NAME}-full-config.cmake"
)
if(EXISTS "${_holoscan_rapids_config}")
  file(COPY_FILE "${_holoscan_rapids_config}" "${_holoscan_full_config}" ONLY_IF_DIFFERENT)
  file(REMOVE "${_holoscan_rapids_config}")
elseif(NOT EXISTS "${_holoscan_full_config}")
  message(FATAL_ERROR
      "RAPIDS generated neither ${_holoscan_rapids_config} nor ${_holoscan_full_config}"
  )
endif()
set(_holoscan_install_package_dir
    "${CMAKE_CURRENT_BINARY_DIR}/install-package/${HOLOSCAN_PACKAGE_NAME}"
)
file(MAKE_DIRECTORY "${_holoscan_install_package_dir}")
set(_holoscan_install_config
    "${_holoscan_install_package_dir}/${HOLOSCAN_PACKAGE_NAME}-config.cmake"
)
set(_holoscan_install_config_version
    "${_holoscan_install_package_dir}/${HOLOSCAN_PACKAGE_NAME}-config-version.cmake"
)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/holoscan-config.cmake.in"
    "${_holoscan_install_config}"
    INSTALL_DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/${HOLOSCAN_PACKAGE_NAME}"
)
install(FILES "${_holoscan_install_config}"
    DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/${HOLOSCAN_PACKAGE_NAME}"
    COMPONENT ${HOLOSCAN_PACKAGE_NAME}
)
install(FILES "${_holoscan_full_config}"
    DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/${HOLOSCAN_PACKAGE_NAME}"
    COMPONENT ${HOLOSCAN_PACKAGE_NAME}
)
if(HOLOSCAN_BUILD_CONFIG)
  install(FILES "${_holoscan_install_config}"
      DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/${HOLOSCAN_PACKAGE_NAME}"
      COMPONENT holoscan-config
  )
endif()
unset(_holoscan_full_config)
unset(_holoscan_rapids_config)
unset(_holoscan_rapids_package_dir)
# HoloMQ and the patched iceoryx targets are built as subprojects and installed through independent
# export sets. CMake cannot represent those local targets as dependencies of a separate build-tree
# export without duplicating the dependency export graph. External consumers use the relocatable
# install package for both shared and static SDK builds.

# Create config version compatible with any newer version of the SDK
# Note: this is to replace the version file from rapids_expert which does not support
#       `AnyNewerVesion`
write_basic_package_version_file(
        "${_holoscan_install_config_version}"
        COMPATIBILITY AnyNewerVersion)

install(FILES "${_holoscan_install_config_version}"
        RENAME "${HOLOSCAN_PACKAGE_NAME}-config-version.cmake"
        DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
        COMPONENT holoscan-core)
if(HOLOSCAN_BUILD_CONFIG)
    install(FILES "${_holoscan_install_config_version}"
            RENAME "${HOLOSCAN_PACKAGE_NAME}-config-version.cmake"
            DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan"
            COMPONENT holoscan-config)
endif()

# The raw build tree is deliberately not an SDK package. Keep both CMake entry points as identical
# fail-fast stubs so version negotiation cannot hide the supported install-tree guidance.
set(_holoscan_raw_build_package_dir "${CMAKE_CURRENT_BINARY_DIR}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/holoscan-build-tree-package-rejected.cmake.in"
    "${_holoscan_raw_build_package_dir}/${HOLOSCAN_PACKAGE_NAME}-config.cmake"
    @ONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/holoscan-build-tree-package-rejected.cmake.in"
    "${_holoscan_raw_build_package_dir}/${HOLOSCAN_PACKAGE_NAME}-config-version.cmake"
    @ONLY
)
unset(_holoscan_raw_build_package_dir)
unset(_holoscan_install_config_version)
unset(_holoscan_install_config)
unset(_holoscan_install_package_dir)

# ##############################################################################
# # Configure CPack packaging
# ##############################################################################
include(HoloscanCPack)
