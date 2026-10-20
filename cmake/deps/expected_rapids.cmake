# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

holoscan_generate_dependency_patch_command(_holoscan_expected_patch_command
    PATCHES
        "${CMAKE_CURRENT_LIST_DIR}/patches/tl-expected/0001-install-tl-expected-for-holoscan-packaging.patch"
        "${CMAKE_CURRENT_LIST_DIR}/patches/tl-expected/0002-export-tl-expected-target-from-the-build-tree.patch"
        "${CMAKE_CURRENT_LIST_DIR}/patches/tl-expected/0003-make-tl-expected-installation-optional.patch")

rapids_cpm_find(tl-expected 1.1.0
    GLOBAL_TARGETS tl::expected
    BUILD_EXPORT_SET "${HOLOSCAN_DEPEND_EXPORT_SET}"
    INSTALL_EXPORT_SET "${HOLOSCAN_DEPEND_EXPORT_SET}"

    CPM_ARGS

    GITHUB_REPOSITORY TartanLlama/expected
    GIT_TAG v1.1.0
    GIT_SHALLOW TRUE
    PATCH_COMMAND ${_holoscan_expected_patch_command}
    EXCLUDE_FROM_ALL ON

    OPTIONS
    "EXPECTED_BUILD_TESTS OFF"
    "EXPECTED_BUILD_PACKAGE OFF"
    "EXPECTED_INSTALL OFF"
)
unset(_holoscan_expected_patch_command)

# The build keeps its pinned source provider. Installed Debian consumers use the compatible
# libexpected-dev 1.1.0 package and its tl-expected CMake config.

# Set 'tl-expected_SOURCE_DIR' with PARENT_SCOPE so that
# root project can use it to include headers
set(tl-expected_SOURCE_DIR ${tl-expected_SOURCE_DIR} PARENT_SCOPE)
