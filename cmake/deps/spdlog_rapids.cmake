# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
set(version 1.10.0)

rapids_cpm_find(spdlog ${version}
    GLOBAL_TARGETS spdlog::spdlog spdlog::spdlog_header_only

    CPM_ARGS

    GITHUB_REPOSITORY gabime/spdlog
    GIT_TAG v${version}
    GIT_SHALLOW TRUE

    OPTIONS
    "SPDLOG_FMT_EXTERNAL_HO ON"
    EXCLUDE_FROM_ALL
)

if(spdlog_ADDED)
    set(spdlog_SOURCE_DIR "${spdlog_SOURCE_DIR}" PARENT_SCOPE)
    set(spdlog_BINARY_DIR "${spdlog_BINARY_DIR}" PARENT_SCOPE)
    set(spdlog_ADDED "${spdlog_ADDED}" PARENT_SCOPE)
    set(spdlog_VERSION ${version} PARENT_SCOPE)
endif()
