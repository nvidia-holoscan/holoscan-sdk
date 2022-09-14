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

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

# nanovg doesn't have versioning, so we provide a fake version (0.1)
rapids_cpm_find(nanovg 0.1
    GLOBAL_TARGETS nanovg

    CPM_ARGS

    GITHUB_REPOSITORY memononen/nanovg
    GIT_TAG 5f65b43
    EXCLUDE_FROM_ALL
)

if(nanovg_ADDED)
    file(GLOB NANOVG_HEADERS
        ${nanovg_SOURCE_DIR}/src/*.h)

    file(GLOB NANOVG_SOURCES
        ${nanovg_SOURCE_DIR}/src/*.c)

    # nanovg is a static library so we do not need to install it.
    add_library(nanovg STATIC ${NANOVG_SOURCES} ${NANOVG_HEADERS})

    target_include_directories(nanovg
        PUBLIC
        $<BUILD_INTERFACE:${nanovg_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include/nanovg>
    )
endif()
