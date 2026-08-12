# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

# Pin to the post-v1.0.0 MatX commit that adds owned DLManagedTensorVersioned
# import/export support. This is needed for Holoscan Tensor DLPack 1.x
# interop validation until a newer MatX release tag is available.
set(HOLOSCAN_MATX_VERSION 1.0.0)
set(HOLOSCAN_MATX_GIT_TAG 968ca84c137d22b6c43b19d186994498a37c8cc7)

rapids_cpm_find(matx ${HOLOSCAN_MATX_VERSION}
    GLOBAL_TARGETS matx
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports

    CPM_ARGS
    GITHUB_REPOSITORY NVIDIA/MatX
    GIT_TAG ${HOLOSCAN_MATX_GIT_TAG}
    GIT_SHALLOW FALSE
    PATCH_COMMAND bash -c "git apply ${CMAKE_CURRENT_LIST_DIR}/patches/matx_install.patch || [ \$? -eq 1 ]"
    EXCLUDE_FROM_ALL
    OPTIONS CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON
)

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
