# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# FlatBuffers dependency setup for schema-backed message envelopes.

include(${rapids-cmake-dir}/cpm/find.cmake)

# Keep the matching headers, generator, static runtime, and package config together. These
# install directory variables are scoped to holoscan_add_dependency's function invocation.
set(CMAKE_INSTALL_INCLUDEDIR "${HOLOSCAN_THIRD_PARTY_INCLUDE_DIR}")
set(CMAKE_INSTALL_LIBDIR "${HOLOSCAN_THIRD_PARTY_INSTALL_LIB_DIR}")
set(CMAKE_INSTALL_BINDIR "${HOLOSCAN_THIRD_PARTY_INSTALL_BIN_DIR}")

set(HOLOSCAN_FLATBUFFERS_VERSION 25.12)
set(HOLOSCAN_FLATBUFFERS_VERSION_EXACT 25.12.19)

if(TARGET flatbuffers AND NOT TARGET flatbuffers::flatbuffers)
  add_library(flatbuffers::flatbuffers ALIAS flatbuffers)
endif()
if(TARGET flatc AND NOT TARGET flatbuffers::flatc)
  add_executable(flatbuffers::flatc ALIAS flatc)
endif()

if(NOT TARGET flatbuffers::flatbuffers OR NOT TARGET flatbuffers::flatc)
  # GCC rejects linking an executable with both -static and a sanitizer.
  # Keep the self-contained flatc tool for ordinary builds, but let the SDK's
  # advertised sanitizer configurations build the generator without fully static
  # linkage so sanitizer validation can reach Holoscan targets.
  set(_holoscan_flatbuffers_static_flatc ON)
  if(HOLOSCAN_ENABLE_GOOGLE_SANITIZER OR HOLOSCAN_ENABLE_THREAD_SANITIZER)
    set(_holoscan_flatbuffers_static_flatc OFF)
  endif()

  rapids_cpm_find(Flatbuffers ${HOLOSCAN_FLATBUFFERS_VERSION}
    GLOBAL_TARGETS flatbuffers::flatbuffers flatbuffers::flatc
    CPM_ARGS
      GITHUB_REPOSITORY google/flatbuffers
      GIT_TAG v${HOLOSCAN_FLATBUFFERS_VERSION_EXACT}
      GIT_SHALLOW TRUE
      EXCLUDE_FROM_ALL
    OPTIONS
      "FLATBUFFERS_BUILD_TESTS OFF"
      "FLATBUFFERS_BUILD_FLATHASH OFF"
      "FLATBUFFERS_BUILD_FLATC ON"
      "FLATBUFFERS_BUILD_FLATLIB ON"
      "FLATBUFFERS_STATIC_FLATC ${_holoscan_flatbuffers_static_flatc}"
      "FLATBUFFERS_STRICT_MODE ON"
  )

  # GCC 13 reports -Wstringop-overflow from libstdc++'s vector::insert implementation.
  # FlatBuffers already classifies this diagnostic as a non-error in strict mode; suppress it
  # completely on the two fetched targets so CDash remains warning-free.
  if(Flatbuffers_ADDED AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(flatbuffers PRIVATE -Wno-stringop-overflow)
    target_compile_options(flatc PRIVATE -Wno-stringop-overflow)
  endif()
endif()

# When built from source via CPM, targets are "flatbuffers" and "flatc" instead
# of "flatbuffers::flatbuffers" and "flatbuffers::flatc". Create namespaced
# aliases so consumers do not need to know whether the dependency was found or
# fetched.
if(TARGET flatbuffers AND NOT TARGET flatbuffers::flatbuffers)
  add_library(flatbuffers::flatbuffers ALIAS flatbuffers)
endif()
if(TARGET flatc AND NOT TARGET flatbuffers::flatc)
  add_executable(flatbuffers::flatc ALIAS flatc)
endif()
if(NOT TARGET flatbuffers::flatbuffers OR NOT TARGET flatbuffers::flatc)
  message(FATAL_ERROR "FlatBuffers provider did not supply runtime and flatc targets")
endif()

# The locally built static runtime is absorbed into the shared holoscan::schema library.
if(TARGET flatbuffers)
  set_target_properties(flatbuffers PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(NOT Flatbuffers_ADDED)
  set(_holoscan_flatbuffers_provider_version)
  foreach(_holoscan_flatbuffers_version_variable IN ITEMS
      Flatbuffers_VERSION FlatBuffers_VERSION flatbuffers_VERSION)
    if(DEFINED ${_holoscan_flatbuffers_version_variable}
       AND NOT "${${_holoscan_flatbuffers_version_variable}}" STREQUAL "")
      set(_holoscan_flatbuffers_provider_version
        "${${_holoscan_flatbuffers_version_variable}}")
      break()
    endif()
  endforeach()
  if(NOT _holoscan_flatbuffers_provider_version)
    message(FATAL_ERROR
      "A predeclared or found FlatBuffers provider must declare its exact version through "
      "Flatbuffers_VERSION, FlatBuffers_VERSION, or flatbuffers_VERSION"
    )
  endif()
  if(NOT _holoscan_flatbuffers_provider_version VERSION_EQUAL
         "${HOLOSCAN_FLATBUFFERS_VERSION_EXACT}")
    message(FATAL_ERROR
      "FlatBuffers provider version ${_holoscan_flatbuffers_provider_version} does not match "
      "required version ${HOLOSCAN_FLATBUFFERS_VERSION_EXACT}"
    )
  endif()
endif()

# holoscan::schema links the static FlatBuffers runtime PRIVATELY (its objects are absorbed into
# libholoscan_schema.so and hidden via --exclude-libs), so FlatBuffers is not part of the holoscan
# export set. Its headers are still part of holoscan::schema's public API surface (the flatc-generated
# headers include "flatbuffers/..."), so install them next to the SDK headers for install-tree
# consumers. Mirrors the fmt dependency treatment.
set(_holoscan_flatbuffers_include_root)
if(Flatbuffers_ADDED)
  set(_holoscan_flatbuffers_include_root "${Flatbuffers_SOURCE_DIR}/include")
else()
  get_target_property(_holoscan_flatbuffers_include_dirs
    flatbuffers::flatbuffers INTERFACE_INCLUDE_DIRECTORIES)
  foreach(_holoscan_flatbuffers_include_dir IN LISTS _holoscan_flatbuffers_include_dirs)
    if(_holoscan_flatbuffers_include_dir MATCHES
       "^\\$<BUILD_INTERFACE:([^>]+)>$")
      set(_holoscan_flatbuffers_include_candidate "${CMAKE_MATCH_1}")
    elseif(IS_ABSOLUTE "${_holoscan_flatbuffers_include_dir}")
      set(_holoscan_flatbuffers_include_candidate "${_holoscan_flatbuffers_include_dir}")
    else()
      continue()
    endif()
    if(EXISTS "${_holoscan_flatbuffers_include_candidate}/flatbuffers/flatbuffers.h")
      set(_holoscan_flatbuffers_include_root "${_holoscan_flatbuffers_include_candidate}")
      break()
    endif()
  endforeach()
endif()
if(NOT _holoscan_flatbuffers_include_root)
  message(FATAL_ERROR
    "FlatBuffers provider does not expose an installable flatbuffers header directory"
  )
endif()
include(GNUInstallDirs)
install(DIRECTORY "${_holoscan_flatbuffers_include_root}/flatbuffers"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT holoscan-dependencies
)
install(PROGRAMS "$<TARGET_FILE:flatbuffers::flatc>"
  DESTINATION "${HOLOSCAN_THIRD_PARTY_INSTALL_BIN_DIR}"
  COMPONENT holoscan-dependencies
)

# FlatBuffers' Kotlin code generators have duplicate symbols in anonymous
# namespaces that break unity builds.
if(TARGET flatc)
  set_target_properties(flatc PROPERTIES UNITY_BUILD OFF)
endif()

set(Flatbuffers_SOURCE_DIR "${Flatbuffers_SOURCE_DIR}" PARENT_SCOPE)
set(Flatbuffers_BINARY_DIR "${Flatbuffers_BINARY_DIR}" PARENT_SCOPE)
set(Flatbuffers_ADDED "${Flatbuffers_ADDED}" PARENT_SCOPE)
set(Flatbuffers_VERSION ${HOLOSCAN_FLATBUFFERS_VERSION_EXACT} PARENT_SCOPE)
unset(_holoscan_flatbuffers_static_flatc)
unset(_holoscan_flatbuffers_include_candidate)
unset(_holoscan_flatbuffers_include_dirs)
unset(_holoscan_flatbuffers_include_root)
unset(_holoscan_flatbuffers_provider_version)
unset(_holoscan_flatbuffers_version_variable)
