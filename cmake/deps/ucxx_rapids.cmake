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

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

set(ucxx_VERSION 0.46.0)
rapids_cpm_find(ucxx ${ucxx_VERSION}
    GLOBAL_TARGETS ucxx
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports

    CPM_ARGS
        GITHUB_REPOSITORY rapidsai/ucxx
        GIT_TAG v0.46.00
        SOURCE_SUBDIR cpp
        PATCH_COMMAND patch -p1 -N -i ${CMAKE_CURRENT_LIST_DIR}/patches/ucxx.patch

    OPTIONS
        "BUILD_TESTS OFF"
        "UCXX_ENABLE_RMM ON"
        "RMM_LOGGING_LEVEL INFO"
)
