# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

set(CMAKE_INSTALL_INCLUDEDIR "${HOLOSCAN_THIRD_PARTY_INCLUDE_DIR}")
set(CMAKE_INSTALL_LIBDIR "${HOLOSCAN_THIRD_PARTY_INSTALL_LIB_DIR}")

holoscan_generate_dependency_patch_command(_holoscan_dlpack_patch_command
    PATCHES
        "${CMAKE_CURRENT_LIST_DIR}/patches/dlpack/0001-export-dlpack-target-from-the-build-tree.patch")

rapids_cpm_find(dlpack 1.3
    GLOBAL_TARGETS dlpack
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    INSTALL_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports

    CPM_ARGS

    GITHUB_REPOSITORY dmlc/dlpack
    GIT_TAG v1.3
    GIT_SHALLOW TRUE
    PATCH_COMMAND ${_holoscan_dlpack_patch_command}
    EXCLUDE_FROM_ALL
)
unset(_holoscan_dlpack_patch_command)

set(_holoscan_dlpack_install_destination "${HOLOSCAN_THIRD_PARTY_INSTALL_LIB_DIR}/cmake/dlpack")
holoscan_configure_dependency_package_config(
    dlpack "${_holoscan_dlpack_install_destination}" _holoscan_dlpack_install_config
    "${HOLOSCAN_THIRD_PARTY_INCLUDE_DIR}")

# dlpack is a PUBLIC compile dependency of holoscan::core (core/domain/tensor.hpp and dlpack.hpp
# include <dlpack/dlpack.h>), so it is registered in the holoscan export set above and the installed
# holoscan-dependencies.cmake will find_dependency(dlpack). The upstream package is configured
# with private install directories, and EXCLUDE_FROM_ALL suppresses its install rules when embedded.
# Install the headers and a minimal config defining the dlpack::dlpack INTERFACE target.
if(dlpack_ADDED)
    install(DIRECTORY "${dlpack_SOURCE_DIR}/include/dlpack"
        DESTINATION "${HOLOSCAN_THIRD_PARTY_INCLUDE_DIR}"
        COMPONENT holoscan-dependencies
    )
    install(FILES
            "${_holoscan_dlpack_install_config}"
            "${CMAKE_CURRENT_LIST_DIR}/configs/dlpack-config-version.cmake"
        DESTINATION "${_holoscan_dlpack_install_destination}"
        COMPONENT holoscan-dependencies
    )
endif()

# Set 'dlpack_SOURCE_DIR' with PARENT_SCOPE so that
# root project can use it to include headers
set(dlpack_SOURCE_DIR ${dlpack_SOURCE_DIR} PARENT_SCOPE)
