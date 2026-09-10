# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

# Pin to latest post-v1.0.0 MatX commit that includes owned
# DLManagedTensorVersioned import/export support and the upstream parent-CPM
# reuse fix.
set(HOLOSCAN_MATX_VERSION 1.0.0)
set(HOLOSCAN_MATX_GIT_TAG 186c72d4794ead29527f45ca4a6f59caa7a08caf)
holoscan_generate_dependency_patch_command(_holoscan_matx_patch_command
    PATCHES
        "${CMAKE_CURRENT_LIST_DIR}/patches/matx_install.patch"
        "${CMAKE_CURRENT_LIST_DIR}/patches/matx_cmp0144_new.patch")

rapids_cpm_find(matx ${HOLOSCAN_MATX_VERSION}
    GLOBAL_TARGETS matx
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports

    CPM_ARGS
    GITHUB_REPOSITORY NVIDIA/MatX
    GIT_TAG ${HOLOSCAN_MATX_GIT_TAG}
    GIT_SHALLOW FALSE
    # Apply HSDK's install/export support and remove MatX's deprecated CMP0144 OLD
    # override; CMP0144 NEW intentionally honors caller-provided uppercase roots.
    PATCH_COMMAND ${_holoscan_matx_patch_command}
    EXCLUDE_FROM_ALL
    OPTIONS CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON
)
unset(_holoscan_matx_patch_command)

if(matx_ADDED)
    # Correct the include directories for MatX target.
    # MatX's own CMakeLists.txt sets install-interface include directories that are incorrect
    # for the Holoscan SDK's 3rdparty directory layout. We clear them and set the correct ones.
    set_target_properties(matx PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
    target_include_directories(matx INTERFACE
        $<BUILD_INTERFACE:${matx_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/3rdparty/matx>
    )

    # The MatX patch enables upstream install/export rules, but those rules are
    # not assigned to the Holoscan dependency component. Re-install the target
    # export and RAPIDS-generated package files into holoscan-dependencies so
    # component installs and packages can still provide find_package(matx).
    install(EXPORT matx-exports
        FILE matx-targets.cmake
        NAMESPACE matx::
        DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/matx"
        COMPONENT holoscan-dependencies
    )

    install(DIRECTORY "${matx_BINARY_DIR}/rapids-cmake/matx/export/matx/"
        DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}/cmake/matx"
        COMPONENT holoscan-dependencies
    )
endif()
