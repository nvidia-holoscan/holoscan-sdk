# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

# Retrieves the magic_enum header-only library.
set(magic_enum_VERSION 0.9.7)
rapids_cpm_find(magic_enum ${magic_enum_VERSION}
    CPM_ARGS
        GITHUB_REPOSITORY Neargye/magic_enum
        GIT_TAG v${magic_enum_VERSION}
        GIT_SHALLOW TRUE
        PATCH_COMMAND bash -c "patch -Np1 -i ${CMAKE_CURRENT_LIST_DIR}/patches/magic_enum.patch || [ \$? -eq 1 ]"
        EXCLUDE_FROM_ALL
)
