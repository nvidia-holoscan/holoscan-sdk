# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

# Retrieves the magic_enum header-only library.
set(magic_enum_VERSION 0.9.7)
rapids_cpm_find(magic_enum ${magic_enum_VERSION}
    GLOBAL_TARGETS magic_enum::magic_enum
    BUILD_EXPORT_SET ${HOLOSCAN_DEPEND_EXPORT_SET}
    INSTALL_EXPORT_SET ${HOLOSCAN_DEPEND_EXPORT_SET}
    CPM_ARGS
        GITHUB_REPOSITORY Neargye/magic_enum
        GIT_TAG v${magic_enum_VERSION}
        GIT_SHALLOW TRUE
        PATCH_COMMAND bash -c "patch -Np1 -i ${CMAKE_CURRENT_LIST_DIR}/patches/magic_enum.patch || [ \$? -eq 1 ]"
        EXCLUDE_FROM_ALL
)

# magic_enum is a PUBLIC compile dependency of holoscan::core (errors.hpp includes
# <magic_enum/magic_enum.hpp>), so it is registered in the holoscan export set above and the
# installed holoscan-dependencies.cmake will find_dependency(magic_enum). EXCLUDE_FROM_ALL suppresses
# magic_enum's own install rules, so install both the headers and a minimal find-package config that
# defines the magic_enum::magic_enum INTERFACE target pointing at those headers.
if(magic_enum_ADDED)
    include(GNUInstallDirs)
    install(DIRECTORY "${magic_enum_SOURCE_DIR}/include/magic_enum"
        DESTINATION include
        COMPONENT ${HOLOSCAN_DEPEND_INSTALL_COMPONENT}
    )
    install(FILES
            "${CMAKE_CURRENT_LIST_DIR}/configs/magic_enum-config.cmake"
            "${CMAKE_CURRENT_LIST_DIR}/configs/magic_enum-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/magic_enum"
        COMPONENT ${HOLOSCAN_DEPEND_INSTALL_COMPONENT}
    )
endif()

unset(_holoscan_magic_enum_export_args)
unset(_holoscan_magic_enum_install_component)
unset(_holoscan_magic_enum_install_lib_dir)
