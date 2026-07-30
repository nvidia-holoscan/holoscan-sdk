# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)

# Reset CMAKE_INSTALL_DEFAULT_COMPONENT_NAME to 'holoscan-modules' so that installation caused
# by modules in this folder will not install into the default component that Holoscan SDK
# uses.
# (CMAKE_INSTALL_DEFAULT_COMPONENT_NAME is set to 'holoscan-core' by the root CMakeLists.txt
# before this folder is processed.)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "holoscan-modules")

include(FetchContent)

FetchContent_Declare(
  holoviz
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/holoviz
)

FetchContent_Declare(
  holoinfer
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/holoinfer
)

FetchContent_Declare(
  holoipc
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/holoipc
)

FetchContent_MakeAvailable(holoviz holoinfer)

if(HOLOSCAN_BUILD_IPC)
  FetchContent_MakeAvailable(holoipc)
endif()
