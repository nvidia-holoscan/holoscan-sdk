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

include(FetchContent)

set(RMM_GITHUB_REPOSITORY "https://github.com/rapidsai/rmm.git")
set(RMM_TAG "v25.10.00")
set(BUILD_TESTS OFF)

FetchContent_Declare(
    rmm
    GIT_REPOSITORY ${RMM_GITHUB_REPOSITORY}
    GIT_TAG        ${RMM_TAG}
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  cpp

    PATCH_COMMAND patch -p1 -N -i ${CMAKE_CURRENT_LIST_DIR}/patches/rmm.patch
    UPDATE_DISCONNECTED TRUE
)
FetchContent_MakeAvailable(rmm)
