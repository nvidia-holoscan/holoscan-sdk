# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

set(fmt_VERSION 12.1.0)
rapids_cpm_find(fmt "${fmt_VERSION}"
    GLOBAL_TARGETS fmt fmt-header-only
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    CPM_ARGS

    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG "${fmt_VERSION}"
    GIT_SHALLOW TRUE
    PATCH_COMMAND bash -c "patch -p1 -N --dry-run -i ${CMAKE_CURRENT_LIST_DIR}/patches/fmt_cuda.patch >/dev/null && patch -p1 -N -i ${CMAKE_CURRENT_LIST_DIR}/patches/fmt_cuda.patch || patch -p1 -R --dry-run -i ${CMAKE_CURRENT_LIST_DIR}/patches/fmt_cuda.patch >/dev/null"

    OPTIONS
    "FMT_INSTALL ON"
    EXCLUDE_FROM_ALL
)

if(fmt_ADDED)
    # Install the headers needed for development with the SDK
    install(DIRECTORY ${fmt_SOURCE_DIR}/include/fmt
        DESTINATION "include"
        COMPONENT "holoscan-dependencies"
        )
endif()
