# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(${rapids-cmake-dir}/cpm/find.cmake)

# reflect-cpp is a private cold-tooling backend. Do not register it in a public Holoscan export set
# or expose rfl types in installed headers. YAML remains owned by the separately packaged yaml-cpp
# adapter so this dependency only supplies its generic JSON representation and typed conversion.
set(reflectcpp_VERSION 0.25.0)
rapids_cpm_find(reflectcpp "${reflectcpp_VERSION}"
    GLOBAL_TARGETS reflectcpp
    CPM_ARGS
        GITHUB_REPOSITORY getml/reflect-cpp
        GIT_TAG "v${reflectcpp_VERSION}"
        GIT_SHALLOW TRUE
    OPTIONS
        "REFLECTCPP_BUILD_TESTS OFF"
        "REFLECTCPP_YAML OFF"
    EXCLUDE_FROM_ALL
)
