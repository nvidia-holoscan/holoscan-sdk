# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_build_tree_package_consumer LANGUAGES CXX)

set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)

set(holoscan_build_tree_root "$ENV{HOLOSCAN_BUILD_TREE_ROOT}")
set(holoscan_find_mode "$ENV{HOLOSCAN_FIND_MODE}")
if(holoscan_build_tree_root STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_BUILD_TREE_ROOT must name the raw SDK build tree")
endif()

set(holoscan_DIR "${holoscan_build_tree_root}")
if(holoscan_find_mode STREQUAL "direct")
    find_package(holoscan CONFIG REQUIRED NO_DEFAULT_PATH)
elseif(holoscan_find_mode STREQUAL "exact_version")
    find_package(holoscan 5.0.0 EXACT CONFIG REQUIRED NO_DEFAULT_PATH)
elseif(holoscan_find_mode STREQUAL "version_range")
    find_package(holoscan 5.0...<6.0 CONFIG REQUIRED NO_DEFAULT_PATH)
elseif(holoscan_find_mode STREQUAL "component")
    find_package(holoscan CONFIG REQUIRED COMPONENTS core NO_DEFAULT_PATH)
else()
    message(FATAL_ERROR "Unknown HOLOSCAN_FIND_MODE: ${holoscan_find_mode}")
endif()

file(WRITE "${CMAKE_BINARY_DIR}/holoscan-package-accepted"
    "The unsupported raw build-tree package was accepted.\n")
message(FATAL_ERROR "The unsupported raw build-tree package was accepted")
