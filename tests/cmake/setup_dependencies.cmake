# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

list(APPEND CMAKE_MESSAGE_CONTEXT "test_deps")

# During dependency acquisition, CMake can otherwise discover package config
# files that were generated into the active build tree before their imported
# targets are complete. For example, a previous configure can leave a partial
# package config under ${CMAKE_BINARY_DIR}; a later rapids_cpm_find() may prefer
# that config over the CPM source it is about to populate, then fail because the
# expected imported targets are not available yet. Ignore the active build tree
# only while acquiring test dependencies, then restore the caller's setting.
set(_HOLOSCAN_TEST_DEPS_SAVED_CMAKE_IGNORE_PREFIX_PATH "${CMAKE_IGNORE_PREFIX_PATH}")
list(APPEND CMAKE_IGNORE_PREFIX_PATH "${CMAKE_BINARY_DIR}")

# Establish the CPM and preset package infrastructure for the project.
rapids_cpm_init()

# Keep clang-tidy focused on project code, not third-party headers/sources.
set(_HOLOSCAN_TEST_DEPS_SAVED_CMAKE_CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY}")
set(CMAKE_CXX_CLANG_TIDY "")

holoscan_add_dependency(gtest_rapids COMPONENT holoscan-dependencies)

set(CMAKE_CXX_CLANG_TIDY "${_HOLOSCAN_TEST_DEPS_SAVED_CMAKE_CXX_CLANG_TIDY}")
unset(_HOLOSCAN_TEST_DEPS_SAVED_CMAKE_CXX_CLANG_TIDY)

set(CMAKE_IGNORE_PREFIX_PATH "${_HOLOSCAN_TEST_DEPS_SAVED_CMAKE_IGNORE_PREFIX_PATH}")
unset(_HOLOSCAN_TEST_DEPS_SAVED_CMAKE_IGNORE_PREFIX_PATH)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
