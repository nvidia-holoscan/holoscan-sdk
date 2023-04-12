# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

rapids_cpm_find(ajantv2 16.2.0
    GLOBAL_TARGETS AJA::ajantv2

    CPM_ARGS

    GITHUB_REPOSITORY nvidia-holoscan/ntv2
    GIT_TAG 1321a8d4c1a8de696c996d05b65e5aa2934f89d1
    OPTIONS
    "AJA_BUILD_APPS OFF"
    "AJA_BUILD_DOCS OFF"
    "AJA_BUILD_DRIVER OFF"
    "AJA_BUILD_LIBS ON"
    "AJA_BUILD_PLUGINS OFF"
    "AJA_BUILD_QA OFF"
    "AJA_BUILD_TESTS OFF"
    "AJA_INSTALL_HEADERS OFF"
    "AJA_INSTALL_SOURCES OFF"
    EXCLUDE_FROM_ALL
)

if(ajantv2_ADDED)
    set_target_properties(ajantv2 PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # ajantv2 is a static library so we do not need to install it.
    add_library(AJA::ajantv2 ALIAS ajantv2)

    # Install the headers needed for development with the SDK
    install(DIRECTORY ${ajantv2_SOURCE_DIR}/ajalibraries
        DESTINATION "include"
        COMPONENT "holoscan-dependencies"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hh"
        )
endif()
