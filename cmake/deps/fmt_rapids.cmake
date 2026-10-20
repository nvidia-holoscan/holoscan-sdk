# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

set(fmt_VERSION 12.2.0)
holoscan_generate_dependency_patch_command(_holoscan_fmt_patch_command
    PATCHES
        "${CMAKE_CURRENT_LIST_DIR}/patches/fmt/0001-backport-fallback-uint128-bitwise-not.patch")
# fmt is a PUBLIC dependency of holoscan::logger (logger.hpp exposes fmt in its logging macros), so it
# is registered in the holoscan export sets: the installed holoscan-dependencies.cmake calls
# find_dependency(fmt) before holoscan-targets.cmake. (core still links fmt PRIVATELY for its .cpp
# implementation.)
rapids_cpm_find(fmt "${fmt_VERSION}"
    GLOBAL_TARGETS fmt fmt-header-only
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    INSTALL_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    CPM_ARGS

    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG "${fmt_VERSION}"
    GIT_SHALLOW TRUE
    # fmt 12.2.0 omitted unary operator~ from its fallback uint128 type. NVCC selects that fallback,
    # causing CUDA translation units to fail in format_hexfloat. Apply the upstream 12.2.1 fix:
    # https://github.com/fmtlib/fmt/pull/4813 (commit 588b3a0f8f6a8bcf2a959cae882d5b2703e86737)
    PATCH_COMMAND ${_holoscan_fmt_patch_command}

    OPTIONS
    "FMT_INSTALL ON"
    EXCLUDE_FROM_ALL
)
unset(_holoscan_fmt_patch_command)

set(_holoscan_fmt_install_destination "${HOLOSCAN_THIRD_PARTY_INSTALL_LIB_DIR}/cmake/fmt")
holoscan_configure_dependency_package_config(
    fmt "${_holoscan_fmt_install_destination}" _holoscan_fmt_install_config
    "${HOLOSCAN_THIRD_PARTY_INCLUDE_DIR}")

if(fmt_ADDED)
    # Install the headers needed for development with the SDK
    install(DIRECTORY ${fmt_SOURCE_DIR}/include/fmt
        DESTINATION "${HOLOSCAN_THIRD_PARTY_INCLUDE_DIR}"
        COMPONENT "holoscan-dependencies"
        )
    # EXCLUDE_FROM_ALL suppresses fmt's own install rules, so install a minimal find-package config
    # that defines the fmt::fmt-header-only INTERFACE target against the installed headers, so the
    # installed find_dependency(fmt) resolves.
    install(FILES
            "${_holoscan_fmt_install_config}"
            "${CMAKE_CURRENT_LIST_DIR}/configs/fmt-config-version.cmake"
        DESTINATION "${_holoscan_fmt_install_destination}"
        COMPONENT "holoscan-dependencies"
        )
endif()
