# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Minimal package config for the header-only use of fmt, installed by the Holoscan SDK. fmt is a
# PUBLIC dependency of holoscan::logger (logger.hpp includes <fmt/format.h>). fmt's upstream CMake
# config is not installed because the CPM package is EXCLUDE_FROM_ALL, so this defines the
# fmt::fmt-header-only INTERFACE target for installed consumers' find_package(holoscan) ->
# find_dependency(fmt). The headers live in the SDK include dir (three levels up from lib*/cmake/fmt,
# which resolves to the install prefix). FMT_HEADER_ONLY matches how the SDK builds against fmt.
if(NOT TARGET fmt::fmt-header-only)
  add_library(fmt::fmt-header-only INTERFACE IMPORTED)
  get_filename_component(_fmt_include "${CMAKE_CURRENT_LIST_DIR}/../../../include" ABSOLUTE)
  set_target_properties(fmt::fmt-header-only PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "FMT_HEADER_ONLY=1"
    INTERFACE_INCLUDE_DIRECTORIES "${_fmt_include}")
  unset(_fmt_include)
endif()
