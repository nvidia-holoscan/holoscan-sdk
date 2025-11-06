# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

rapids_cpm_find(matx 0.9.3
    GLOBAL_TARGETS matx
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports

    CPM_ARGS
    GITHUB_REPOSITORY NVIDIA/MatX
    GIT_TAG v0.9.3
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)

if(matx_ADDED)
    # Correct the include directories for MatX target.
    # MatX's own CMakeLists.txt sets install-interface include directories that are incorrect
    # for the Holoscan SDK's 3rdparty directory layout. We clear them and set the correct ones.
    set_target_properties(matx PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
    target_include_directories(matx INTERFACE
        $<BUILD_INTERFACE:${matx_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/3rdparty/matx>
    )

    # Install the headers needed for development with the SDK
    # Note: MatX's umbrella header is located in include/matx.h along with 'matx' folder having
    # the actual implementation.
    install(DIRECTORY ${matx_SOURCE_DIR}/include/
        DESTINATION include/3rdparty/matx
        COMPONENT "holoscan-dependencies"
        )
endif()
