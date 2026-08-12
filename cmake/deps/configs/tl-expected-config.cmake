# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Minimal package config for the header-only tl-expected library, installed by the Holoscan SDK.
# tl-expected is a PUBLIC compile dependency of holoscan::core (expected.hpp includes
# <tl/expected.hpp>). Its upstream CMake config is not installed because the CPM package is
# EXCLUDE_FROM_ALL, so this defines the tl::expected INTERFACE target for installed consumers'
# find_package(holoscan) -> find_dependency(tl-expected). The headers live in the SDK include dir
# (three levels up from lib*/cmake/tl-expected, which resolves to the install prefix).
if(NOT TARGET tl::expected)
  add_library(tl::expected INTERFACE IMPORTED)
  get_filename_component(_tl_expected_include "${CMAKE_CURRENT_LIST_DIR}/../../../include" ABSOLUTE)
  set_target_properties(tl::expected PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_tl_expected_include}")
  unset(_tl_expected_include)
endif()
