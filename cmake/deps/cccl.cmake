# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# CCCL is a dependency of MatX and RMM.

# Try to find system CCCL in the build container first.
# Subsequent RAPIDS CPM components (MatX) can get confused
# if both a compatible system and cache CCCL are present.
find_package(CCCL 3.0.3 PATHS /usr/local/cuda/lib64/cmake QUIET)
if(NOT CCCL_FOUND)
    rapids_cpm_find(CCCL 3.0.3
        GLOBAL_TARGETS CCCL CCCL::CCCL CCCL::CUB CCCL::libcudacxx
        BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports

        CPM_ARGS
        # CPM fails to cache CCCL in the expected CPM source directory, so use
        # the default FetchContent location for caching instead
        # https://gitlab.kitware.com/cmake/cmake/-/issues/25714
        SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/cccl-src

        GIT_SHALLOW TRUE

        EXCLUDE_FROM_ALL
        OPTIONS "CCCL_TOPLEVEL_PROJECT OFF"
                "CCCL_ENABLE_INSTALL_RULES ON"
    )
endif()
