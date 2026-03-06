# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

set(nvtx_VERSION 3.3)
rapids_cpm_find(nvtx3 ${nvtx_VERSION}
    GLOBAL_TARGETS nvtx3-c nvtx3-cpp
    CPM_ARGS
        GITHUB_REPOSITORY NVIDIA/NVTX
        GIT_TAG v3.3.0-c-cpp
        GIT_SHALLOW TRUE
        PATCH_COMMAND patch -p1 -N -i ${CMAKE_CURRENT_LIST_DIR}/patches/nvtx3.patch
        EXCLUDE_FROM_ALL

    OPTIONS
        NVTX3_INSTALL ON
)
