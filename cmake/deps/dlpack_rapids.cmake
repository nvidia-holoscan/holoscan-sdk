# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

holoscan_generate_dependency_patch_command(_holoscan_dlpack_patch_command
    PATCHES "${CMAKE_CURRENT_LIST_DIR}/patches/dlpack.patch")

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

# dlpack is a PUBLIC compile dependency of holoscan::core (core/domain/tensor.hpp and dlpack.hpp
# include <dlpack/dlpack.h>), so it is registered in the holoscan export set above and the installed
# holoscan-dependencies.cmake will find_dependency(dlpack). EXCLUDE_FROM_ALL suppresses dlpack's own
# install rules, so install both the headers and a minimal find-package config that defines the
# dlpack::dlpack INTERFACE target pointing at those headers.
if(dlpack_ADDED)
    install(DIRECTORY "${dlpack_SOURCE_DIR}/include/dlpack"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT holoscan-dependencies
    )
    install(FILES
            "${CMAKE_CURRENT_LIST_DIR}/configs/dlpack-config.cmake"
            "${CMAKE_CURRENT_LIST_DIR}/configs/dlpack-config-version.cmake"
        DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/dlpack"
        COMPONENT holoscan-dependencies
    )
endif()

# Set 'dlpack_SOURCE_DIR' with PARENT_SCOPE so that
# root project can use it to include headers
set(dlpack_SOURCE_DIR ${dlpack_SOURCE_DIR} PARENT_SCOPE)
