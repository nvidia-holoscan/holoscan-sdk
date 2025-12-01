# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

rapids_cpm_find(rmm 25.10.00
    GLOBAL_TARGETS rmm
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    CPM_ARGS

    GITHUB_REPOSITORY rapidsai/rmm
    GIT_TAG v25.10.00
    SOURCE_SUBDIR cpp
    GIT_SHALLOW TRUE
    OPTIONS
       BUILD_TESTS OFF
    EXCLUDE_FROM_ALL
)

if(rmm_ADDED)
    # Install the headers needed for development with the SDK
    install(DIRECTORY ${rmm_SOURCE_DIR}/cpp/include/rmm
        DESTINATION "include"
        COMPONENT "holoscan-dependencies"
        )
    install(DIRECTORY ${rmm_BINARY_DIR}/include/rmm
        DESTINATION "include"
        COMPONENT "holoscan-dependencies"
        )
    install(DIRECTORY ${CPM_PACKAGE_rapids_logger_SOURCE_DIR}/include/rapids_logger
        DESTINATION "include"
        COMPONENT "holoscan-dependencies"
        )
    install(
       TARGETS rmm rapids_logger
       DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}"
       COMPONENT "holoscan-dependencies"
    )
endif()
