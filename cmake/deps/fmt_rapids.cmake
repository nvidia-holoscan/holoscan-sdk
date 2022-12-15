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

rapids_cpm_find(fmt 8.1.1
    GLOBAL_TARGETS fmt fmt-header-only
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    CPM_ARGS

    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG 8.1.1
    GIT_SHALLOW TRUE

    OPTIONS
    "FMT_INSTALL ON"
    EXCLUDE_FROM_ALL
)

if(fmt_ADDED)
    # Install the headers needed for development with the SDK
    install(DIRECTORY ${fmt_SOURCE_DIR}/include/fmt
        DESTINATION "include"
        COMPONENT "holoscan-dependencies"
        )
endif()
