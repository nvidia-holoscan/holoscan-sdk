# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Minimal package config for the header-only magic_enum library, installed by the Holoscan SDK.
# magic_enum is a PUBLIC compile dependency of holoscan::core (errors.hpp includes
# <magic_enum/magic_enum.hpp>). Its upstream CMake config is not installed because the CPM package is
# EXCLUDE_FROM_ALL, so this defines the INTERFACE target for installed consumers' find_package(holoscan)
# -> find_dependency(magic_enum). The headers live in the SDK include dir (three levels up from
# lib*/cmake/magic_enum, which resolves to the install prefix regardless of lib vs lib64).
if(NOT TARGET magic_enum::magic_enum)
  add_library(magic_enum::magic_enum INTERFACE IMPORTED)
  get_filename_component(_magic_enum_include "${CMAKE_CURRENT_LIST_DIR}/../../../include" ABSOLUTE)
  set_target_properties(magic_enum::magic_enum PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_magic_enum_include}")
  unset(_magic_enum_include)
endif()
