# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

# Retrieves the magic_enum header-only library.
set(magic_enum_VERSION 0.9.7)
holoscan_generate_dependency_patch_command(_holoscan_magic_enum_patch_command
    PATCHES
        "${CMAKE_CURRENT_LIST_DIR}/patches/magic_enum/0001-install-magic-enum-for-holoscan-packaging.patch")
rapids_cpm_find(magic_enum ${magic_enum_VERSION}
    GLOBAL_TARGETS magic_enum::magic_enum
    BUILD_EXPORT_SET ${HOLOSCAN_DEPEND_EXPORT_SET}
    INSTALL_EXPORT_SET ${HOLOSCAN_DEPEND_EXPORT_SET}
    CPM_ARGS
        GITHUB_REPOSITORY Neargye/magic_enum
        GIT_TAG v${magic_enum_VERSION}
        GIT_SHALLOW TRUE
        PATCH_COMMAND ${_holoscan_magic_enum_patch_command}
        EXCLUDE_FROM_ALL
)
unset(_holoscan_magic_enum_patch_command)

set(_holoscan_magic_enum_install_destination "${CMAKE_INSTALL_LIBDIR}/cmake/magic_enum")
holoscan_configure_dependency_package_config(
    magic_enum
    "${_holoscan_magic_enum_install_destination}"
    _holoscan_magic_enum_install_config)

# magic_enum is a private compiled dependency of holoscan::core. Static SDK exports retain private
# libraries as link-only requirements, so the installed dependency closure still provides the
# magic_enum target. EXCLUDE_FROM_ALL suppresses magic_enum's own install rules; install both the
# headers and a minimal find-package config that defines the magic_enum::magic_enum INTERFACE target.
if(magic_enum_ADDED)
    include(GNUInstallDirs)
    install(DIRECTORY "${magic_enum_SOURCE_DIR}/include/magic_enum"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT ${HOLOSCAN_DEPEND_INSTALL_COMPONENT}
    )
    install(FILES
            "${_holoscan_magic_enum_install_config}"
            "${CMAKE_CURRENT_LIST_DIR}/configs/magic_enum-config-version.cmake"
        DESTINATION "${_holoscan_magic_enum_install_destination}"
        COMPONENT ${HOLOSCAN_DEPEND_INSTALL_COMPONENT}
    )
endif()

unset(_holoscan_magic_enum_export_args)
unset(_holoscan_magic_enum_install_component)
unset(_holoscan_magic_enum_install_lib_dir)
