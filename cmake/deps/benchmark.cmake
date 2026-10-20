# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(TARGET benchmark::benchmark AND TARGET benchmark::benchmark_main)
    return()
endif()

set(version 1.9.5)
rapids_cpm_find(benchmark ${version}
    GLOBAL_TARGETS benchmark::benchmark benchmark::benchmark_main
    CPM_ARGS FIND_PACKAGE_ARGUMENTS "EXACT"
    GIT_REPOSITORY "https://github.com/google/benchmark.git"
    GIT_TAG v${version}
    GIT_SHALLOW ON
    PATCH_COMMAND ""
    EXCLUDE_FROM_ALL OFF
    OPTIONS
        "BENCHMARK_ENABLE_TESTING OFF"
        "BENCHMARK_ENABLE_GTEST_TESTS OFF"
        "BENCHMARK_ENABLE_INSTALL OFF"
        "BENCHMARK_INSTALL_DOCS OFF"
        "BENCHMARK_ENABLE_WERROR OFF"
)

set(benchmark_SOURCE_DIR "${benchmark_SOURCE_DIR}" PARENT_SCOPE)
set(benchmark_BINARY_DIR "${benchmark_BINARY_DIR}" PARENT_SCOPE)
set(benchmark_ADDED "${benchmark_ADDED}" PARENT_SCOPE)
set(benchmark_VERSION ${version} PARENT_SCOPE)
