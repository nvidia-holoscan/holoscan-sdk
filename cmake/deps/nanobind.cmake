# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
if(COMMAND nanobind_add_module)
    return()
endif()

include(${rapids-cmake-dir}/cpm/find.cmake)

set(nanobind_VERSION 3.0.1)
rapids_cpm_find(nanobind ${nanobind_VERSION}
    CPM_ARGS
        GITHUB_REPOSITORY wjakob/nanobind
        GIT_TAG v${nanobind_VERSION}
        GIT_SHALLOW TRUE
        GIT_SUBMODULES_RECURSE TRUE
        EXCLUDE_FROM_ALL

    OPTIONS
        "NB_CREATE_INSTALL_RULES OFF"
        "NB_TEST OFF"
)
