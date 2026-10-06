# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(${rapids-cmake-dir}/cpm/find.cmake)

# yaml-cpp is a private implementation provider for both config_io and inference.
# Find an existing version-compatible package or else fetch and build from source.
set(YAML_CPP_VERSION 0.9.0)
rapids_cpm_find(yaml-cpp ${YAML_CPP_VERSION}
    GLOBAL_TARGETS yaml-cpp yaml-cpp::yaml-cpp
    CPM_ARGS
        GITHUB_REPOSITORY jbeder/yaml-cpp
        GIT_TAG yaml-cpp-${YAML_CPP_VERSION}
        GIT_SHALLOW TRUE
    OPTIONS
        "YAML_CPP_BUILD_CONTRIB OFF"
        "YAML_CPP_BUILD_TESTS OFF"
        "YAML_CPP_BUILD_TOOLS OFF"
        "YAML_BUILD_SHARED_LIBS OFF"
        "YAML_ENABLE_PIC ON"
        "YAML_CPP_INSTALL OFF"
    EXCLUDE_FROM_ALL
)

# yaml-cpp builds from source as the unqualified target while its installed package provides the
# canonical namespaced target. Normalize either provider.
if(TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
  add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
endif()
if(NOT TARGET yaml-cpp::yaml-cpp)
  message(FATAL_ERROR
      "yaml-cpp ${YAML_CPP_VERSION} must provide the yaml-cpp::yaml-cpp target"
  )
endif()

# This provider is absorbed into Holoscan shared libraries and repackaged privately for static
# config_io consumers, so it must be a static PIC archive regardless of its acquisition source.
get_target_property(_holoscan_yaml_cpp_target_type yaml-cpp::yaml-cpp TYPE)
if(NOT _holoscan_yaml_cpp_target_type STREQUAL "STATIC_LIBRARY")
  message(FATAL_ERROR
      "yaml-cpp ${YAML_CPP_VERSION} provider must supply yaml-cpp::yaml-cpp as a static library; "
      "got ${_holoscan_yaml_cpp_target_type}"
  )
endif()
unset(_holoscan_yaml_cpp_target_type)
if(TARGET yaml-cpp)
  set_target_properties(yaml-cpp PROPERTIES
      CXX_CLANG_TIDY ""
  )
endif()
