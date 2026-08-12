# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Minimal package config for the header-only dlpack library, installed by the Holoscan SDK. dlpack is
# a PUBLIC compile dependency of holoscan::core (core/domain/tensor.hpp and dlpack.hpp include
# <dlpack/dlpack.h>). Its upstream CMake config is not installed because the CPM package is
# EXCLUDE_FROM_ALL, so this defines the dlpack::dlpack INTERFACE target for installed consumers'
# find_package(holoscan) -> find_dependency(dlpack). The headers live in the SDK include dir (three
# levels up from lib*/cmake/dlpack, which resolves to the install prefix).
if(NOT TARGET dlpack::dlpack)
  add_library(dlpack::dlpack INTERFACE IMPORTED)
  get_filename_component(_dlpack_include "${CMAKE_CURRENT_LIST_DIR}/../../../include" ABSOLUTE)
  set_target_properties(dlpack::dlpack PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_dlpack_include}")
  unset(_dlpack_include)
endif()
