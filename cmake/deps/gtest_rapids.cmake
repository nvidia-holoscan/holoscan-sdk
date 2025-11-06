# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Derived from https://docs.rapids.ai/api/rapids-cmake/stable/packages/rapids_cpm_gtest
#
# Workaround for issue where `FetchContent_Declare(GTest)` does not respect `SOURCE_DIR` for caching.
# https://github.com/google/googletest/issues/4384
# https://gitlab.kitware.com/cmake/cmake/-/issues/25714
set(version 1.17.0)
rapids_cpm_find(googletest ${version}
    GLOBAL_TARGETS GTest::gtest GTest::gmock GTest::gtest_main GTest::gmock_main
    CPM_ARGS FIND_PACKAGE_ARGUMENTS "EXACT"
    GIT_REPOSITORY "https://github.com/google/googletest.git"
    GIT_TAG v${version}
    GIT_SHALLOW ON
    PATCH_COMMAND ""
    EXCLUDE_FROM_ALL OFF
    OPTIONS "INSTALL_GTEST OFF"
)

# Propagate up variables that CPMFindPackage provide
set(googletest_SOURCE_DIR "${googletest_SOURCE_DIR}" PARENT_SCOPE)
set(googletest_BINARY_DIR "${googletest_BINARY_DIR}" PARENT_SCOPE)
set(googletest_ADDED "${googletest_ADDED}" PARENT_SCOPE)
set(googletest_VERSION ${version} PARENT_SCOPE)

if(TARGET GTest::gtest AND NOT TARGET GTest::gmock)
    message(WARNING "The googletest package found doesn't provide gmock. If you run into 'GTest::gmock target not found' issues you need to use a different version of GTest.The easiest way is to request building GTest from source by adding the following to the cmake invocation:
    '-DCPM_DOWNLOAD_googletest=ON'")
endif()

if(NOT TARGET GTest::gtest AND TARGET gtest)
    add_library(GTest::gtest ALIAS gtest)
    add_library(GTest::gtest_main ALIAS gtest_main)
endif()

if(NOT TARGET GTest::gmock AND TARGET gmock)
    add_library(GTest::gmock ALIAS gmock)
    add_library(GTest::gmock_main ALIAS gmock_main)
endif()
