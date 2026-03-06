# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(${rapids-cmake-dir}/cpm/find.cmake)

rapids_cpm_find(concurrent_queue 1.0.4
    GLOBAL_TARGETS concurrent_queue
    CPM_ARGS

    GITHUB_REPOSITORY cameron314/concurrentqueue
    GIT_TAG v1.0.4
    GIT_SHALLOW TRUE
    PATCH_COMMAND patch -p1 -N -i ${CMAKE_CURRENT_LIST_DIR}/patches/concurrentqueue.patch

    EXCLUDE_FROM_ALL
)

# Set 'concurrent_queue_SOURCE_DIR' with PARENT_SCOPE so that
# root project can use it to include headers
set(concurrent_queue_SOURCE_DIR ${concurrent_queue_SOURCE_DIR} PARENT_SCOPE)
