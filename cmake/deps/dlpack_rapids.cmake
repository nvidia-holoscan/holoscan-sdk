# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

rapids_cpm_find(dlpack 0.7
    GLOBAL_TARGETS dlpack

    CPM_ARGS

    GITHUB_REPOSITORY dmlc/dlpack
    GIT_TAG v0.7
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)

# Set 'dlpack_SOURCE_DIR' with PARENT_SCOPE so that
# root project can use it to include headers
set(dlpack_SOURCE_DIR ${dlpack_SOURCE_DIR} PARENT_SCOPE)
