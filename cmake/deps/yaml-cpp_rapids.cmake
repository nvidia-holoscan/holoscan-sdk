# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(${rapids-cmake-dir}/cpm/find.cmake)

# yaml-cpp is a private config_io implementation provider. Fetch the pinned source instead of
# requiring a system package so clean SDK builds and installed-package validation use the same
# implementation. Its headers, upstream archive name, package configuration, and target remain
# outside Holoscan's public install surface; static config_io packaging copies only a renamed
# Holoscan-private archive from the owning component.
if(TARGET yaml-cpp OR TARGET yaml-cpp::yaml-cpp)
  message(FATAL_ERROR
      "Holoscan config_io requires its pinned private yaml-cpp provider, but a yaml-cpp target "
      "already exists. Remove the pre-existing yaml-cpp target before adding Holoscan so RAPIDS "
      "CPM cannot silently reuse an unverified provider."
  )
endif()

set(YAML_CPP_VERSION 0.9.0)
set(_holoscan_yaml_cpp_saved_download "${CPM_DOWNLOAD_yaml-cpp}")
set(_holoscan_yaml_cpp_had_download FALSE)
if(DEFINED CPM_DOWNLOAD_yaml-cpp)
  set(_holoscan_yaml_cpp_had_download TRUE)
endif()
set(CPM_DOWNLOAD_yaml-cpp ON)
rapids_cpm_find(yaml-cpp ${YAML_CPP_VERSION}
    GLOBAL_TARGETS yaml-cpp
    CPM_ARGS
        GITHUB_REPOSITORY jbeder/yaml-cpp
        GIT_TAG yaml-cpp-${YAML_CPP_VERSION}
        GIT_SHALLOW TRUE
    OPTIONS
        "YAML_CPP_BUILD_CONTRIB OFF"
        "YAML_CPP_BUILD_TESTS OFF"
        "YAML_CPP_BUILD_TOOLS OFF"
        "YAML_BUILD_SHARED_LIBS OFF"
        "YAML_CPP_INSTALL OFF"
    EXCLUDE_FROM_ALL
)
if(_holoscan_yaml_cpp_had_download)
  set(CPM_DOWNLOAD_yaml-cpp "${_holoscan_yaml_cpp_saved_download}")
else()
  unset(CPM_DOWNLOAD_yaml-cpp)
endif()
unset(_holoscan_yaml_cpp_had_download)
unset(_holoscan_yaml_cpp_saved_download)

if(NOT TARGET yaml-cpp)
  message(FATAL_ERROR "The private config_io yaml-cpp provider was not created")
endif()
set_target_properties(yaml-cpp PROPERTIES
    CXX_CLANG_TIDY ""
    POSITION_INDEPENDENT_CODE ON
)
if(NOT TARGET yaml-cpp::yaml-cpp)
  add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
endif()
