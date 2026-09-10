# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

list(APPEND CMAKE_MESSAGE_CONTEXT "benchmark_deps")

# Ignore build-tree package configs while resolving the Google Benchmark dependency.
set(_HOLOSCAN_BENCHMARK_DEPS_SAVED_CMAKE_IGNORE_PREFIX_PATH "${CMAKE_IGNORE_PREFIX_PATH}")

# Use holoscan_BINARY_DIR when embedded and CMAKE_BINARY_DIR otherwise.
if(DEFINED holoscan_BINARY_DIR)
    list(APPEND CMAKE_IGNORE_PREFIX_PATH "${holoscan_BINARY_DIR}")
else()
    list(APPEND CMAKE_IGNORE_PREFIX_PATH "${CMAKE_BINARY_DIR}")
endif()

# SDK builds acquire benchmark through the shared dependency module in
# public/cmake/deps/benchmark.cmake via holoscan_add_dependency(). Standalone benchmark
# builds skip that path and use the find_package/FetchContent_MakeAvailable
# fallback below.
if(COMMAND holoscan_add_dependency)
    rapids_cpm_init()

    set(_HOLOSCAN_BENCHMARK_DEPS_SAVED_CMAKE_CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY}")
    set(CMAKE_CXX_CLANG_TIDY "")

    holoscan_add_dependency(benchmark COMPONENT holoscan-dependencies)

    set(CMAKE_CXX_CLANG_TIDY "${_HOLOSCAN_BENCHMARK_DEPS_SAVED_CMAKE_CXX_CLANG_TIDY}")
    unset(_HOLOSCAN_BENCHMARK_DEPS_SAVED_CMAKE_CXX_CLANG_TIDY)
endif()

set(CMAKE_IGNORE_PREFIX_PATH "${_HOLOSCAN_BENCHMARK_DEPS_SAVED_CMAKE_IGNORE_PREFIX_PATH}")
unset(_HOLOSCAN_BENCHMARK_DEPS_SAVED_CMAKE_IGNORE_PREFIX_PATH)

if(NOT TARGET benchmark::benchmark_main)
    find_package(benchmark 1.9.5 CONFIG QUIET)
endif()

# Installed benchmark sources are buildable without the SDK dependency helper.
# Prefer a system package, then fetch Google Benchmark directly for that
# standalone path.
if(NOT TARGET benchmark::benchmark_main)
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5
        GIT_SHALLOW ON
    )
    FetchContent_MakeAvailable(benchmark)
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
