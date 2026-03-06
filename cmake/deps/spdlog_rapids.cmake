# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

# Here we are using rapids_cpm_find() function instead of rapids_cpm_spdlog() function
# (https://docs.rapids.ai/api/rapids-cmake/stable/packages/rapids_cpm_spdlog.html), to
# override the default options.
set(version 1.14.1)

include(FetchContent)

set(SPDLOG_GITHUB_REPOSITORY "https://github.com/gabime/spdlog.git")
set(SPDLOG_TAG "v${version}")

set(SPDLOG_FMT_EXTERNAL_HO ON)   # header-only
set(SPDLOG_FMT_EXTERNAL OFF)
set(SPDLOG_INSTALL ON)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY ${SPDLOG_GITHUB_REPOSITORY}
    GIT_TAG        ${SPDLOG_TAG}
    GIT_SHALLOW    TRUE
    PATCH_COMMAND patch -p1 -i ${CMAKE_CURRENT_LIST_DIR}/patches/spdlog.patch
    UPDATE_DISCONNECTED TRUE
)
FetchContent_MakeAvailable(spdlog)

if(spdlog_ADDED)
    set(spdlog_SOURCE_DIR "${spdlog_SOURCE_DIR}" PARENT_SCOPE)
    set(spdlog_BINARY_DIR "${spdlog_BINARY_DIR}" PARENT_SCOPE)
    set(spdlog_ADDED "${spdlog_ADDED}" PARENT_SCOPE)
    set(spdlog_VERSION ${version} PARENT_SCOPE)
endif()
