# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(HOLOSCAN_SLANG_VERSION 2026.14)
set(HOLOVIZ_SLANG_PROVIDER "AUTO" CACHE STRING
  "Slang provider: AUTO, PREBUILT, SOURCE, or EXTERNAL")
set_property(CACHE HOLOVIZ_SLANG_PROVIDER PROPERTY STRINGS AUTO PREBUILT SOURCE EXTERNAL)
string(TOUPPER "${HOLOVIZ_SLANG_PROVIDER}" _holoviz_slang_provider)
set(_holoviz_slang_providers AUTO PREBUILT SOURCE EXTERNAL)
if(NOT _holoviz_slang_provider IN_LIST _holoviz_slang_providers)
  message(FATAL_ERROR
    "Unknown HOLOVIZ_SLANG_PROVIDER=${HOLOVIZ_SLANG_PROVIDER}; "
    "expected AUTO, PREBUILT, SOURCE, or EXTERNAL")
endif()
unset(_holoviz_slang_providers)

if(TARGET slang::slang)
  unset(_holoviz_slang_provider)
  return()
endif()

# External mode makes runtime ownership explicit while retaining CMake's standard package search
# controls. A package-specific root takes precedence over a cached slang_DIR, which might
# otherwise be a stale result from a previous bundled configure. Without a root, slang_DIR and
# CMAKE_PREFIX_PATH retain their normal find_package semantics. The supplier owns deployment of
# the found runtime.
if(_holoviz_slang_provider STREQUAL "EXTERNAL")
  if(slang_ROOT)
    unset(slang_DIR CACHE)
    find_package(slang ${HOLOSCAN_SLANG_VERSION} CONFIG REQUIRED
      PATHS "${slang_ROOT}"
      NO_DEFAULT_PATH
    )
  else()
    find_package(slang ${HOLOSCAN_SLANG_VERSION} CONFIG REQUIRED)
  endif()
  unset(_holoviz_slang_provider)
  return()
endif()

# Keep this recipe self-contained. RAPIDS guards this command module against repeated inclusion;
# the explicit include also lets configure-contract tests replace acquisition with a local stub.
include(${rapids-cmake-dir}/cpm/find.cmake)

if(DEFINED HOLOSCAN_DEPEND_INSTALL_COMPONENT)
  set(_holoviz_slang_install_component "${HOLOSCAN_DEPEND_INSTALL_COMPONENT}")
elseif(DEFINED HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT)
  set(_holoviz_slang_install_component "${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}")
else()
  message(FATAL_ERROR "Holoscan Visualization Slang dependency has no install component")
endif()

# Both bundled lanes own their selected Slang and its install rules. Force RAPIDS CPM acquisition
# so an unrelated package from CMAKE_PREFIX_PATH or a prior find_package cache cannot substitute
# for that runtime, then restore the caller's override after the recipe completes.
set(_holoviz_slang_saved_download "${CPM_DOWNLOAD_slang}")
set(_holoviz_slang_download_defined FALSE)
if(DEFINED CPM_DOWNLOAD_slang)
  set(_holoviz_slang_download_defined TRUE)
endif()
set(CPM_DOWNLOAD_slang ON)

if(_holoviz_slang_provider STREQUAL "AUTO" OR
    _holoviz_slang_provider STREQUAL "PREBUILT")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_holoviz_slang_os "linux")
    set(_holoviz_slang_ext "tar.gz")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_holoviz_slang_os "windows")
    set(_holoviz_slang_ext "zip")
  else()
    set(_holoviz_slang_os "")
  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(_holoviz_slang_arch "aarch64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(_holoviz_slang_arch "x86_64")
  else()
    set(_holoviz_slang_arch "")
  endif()

  # The prebuilt binaries ship in the release package, so each asset is pinned by hash.
  set(_holoviz_slang_sha256_linux_aarch64
    "7acbfe80260009263346f72efaadabfa7666248a045e720b573ba78fb008ac42")
  set(_holoviz_slang_sha256_linux_x86_64
    "9445cf0b35332bfda055c9424e307f957ff58befb04c5b45b9f9f29894a41cf8")
  set(_holoviz_slang_sha256_windows_aarch64
    "4acec8a0aedd574043a37aa0fdcaed11b08d7f26399b09166fcacc4f60cb73a2")
  set(_holoviz_slang_sha256_windows_x86_64
    "36029c50ef0c82f2616ffb02e0ed27d642cb44a2a297d531cc2ad333b85b85b6")

  if(_holoviz_slang_os AND _holoviz_slang_arch)
    set(_holoviz_slang_sha256
      "${_holoviz_slang_sha256_${_holoviz_slang_os}_${_holoviz_slang_arch}}")
    rapids_cpm_find(slang ${HOLOSCAN_SLANG_VERSION}
      CPM_ARGS
        URL "https://github.com/shader-slang/slang/releases/download/v${HOLOSCAN_SLANG_VERSION}/slang-${HOLOSCAN_SLANG_VERSION}-${_holoviz_slang_os}-${_holoviz_slang_arch}.${_holoviz_slang_ext}"
        URL_HASH SHA256=${_holoviz_slang_sha256}
        DOWNLOAD_ONLY ON
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    unset(slang_DIR CACHE)
    find_package(slang ${HOLOSCAN_SLANG_VERSION} CONFIG REQUIRED
      HINTS
        "${slang_SOURCE_DIR}/lib/cmake/slang"
        "${slang_SOURCE_DIR}/lib64/cmake/slang"
        "${slang_SOURCE_DIR}/cmake"
      NO_DEFAULT_PATH
    )
    unset(slang_DIR CACHE)

    if(WIN32)
      install(DIRECTORY "${slang_SOURCE_DIR}/bin/"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${_holoviz_slang_install_component}
        FILES_MATCHING PATTERN "*.dll"
      )
    else()
      install(DIRECTORY "${slang_SOURCE_DIR}/lib/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT ${_holoviz_slang_install_component}
        FILES_MATCHING PATTERN "*.so*"
      )
    endif()
  else()
    if(_holoviz_slang_provider STREQUAL "PREBUILT")
      message(FATAL_ERROR
        "No prebuilt Slang release for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}; "
        "select HOLOVIZ_SLANG_PROVIDER=SOURCE or EXTERNAL")
    else()
      message(STATUS
        "No prebuilt Slang release for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}; "
        "falling back to source build")
      set(_holoviz_slang_provider SOURCE)
    endif()
  endif()

  unset(_holoviz_slang_arch)
  unset(_holoviz_slang_os)
  unset(_holoviz_slang_ext)
  unset(_holoviz_slang_sha256)
  unset(_holoviz_slang_sha256_linux_aarch64)
  unset(_holoviz_slang_sha256_linux_x86_64)
  unset(_holoviz_slang_sha256_windows_aarch64)
  unset(_holoviz_slang_sha256_windows_x86_64)
endif()

if(NOT TARGET slang::slang AND _holoviz_slang_provider STREQUAL "SOURCE")
  set(SLANG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_GFX OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_SLANGD OFF CACHE BOOL "" FORCE)
  set(SLANG_ENABLE_SLANG_RHI OFF CACHE BOOL "" FORCE)
  rapids_cpm_find(slang ${HOLOSCAN_SLANG_VERSION}
    GLOBAL_TARGETS slang
    CPM_ARGS
      GITHUB_REPOSITORY shader-slang/slang
      GIT_TAG v${HOLOSCAN_SLANG_VERSION}
      GIT_SHALLOW TRUE
      EXCLUDE_FROM_ALL
  )
  unset(slang_DIR CACHE)

  if(NOT TARGET slang::slang AND TARGET slang)
    add_library(slang::slang ALIAS slang)
  endif()

  install(TARGETS slang
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      COMPONENT ${_holoviz_slang_install_component}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      COMPONENT ${_holoviz_slang_install_component}
  )
endif()

if(_holoviz_slang_download_defined)
  set(CPM_DOWNLOAD_slang "${_holoviz_slang_saved_download}")
else()
  unset(CPM_DOWNLOAD_slang)
endif()

unset(_holoviz_slang_download_defined)
unset(_holoviz_slang_install_component)
unset(_holoviz_slang_provider)
unset(_holoviz_slang_saved_download)
