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

rapids_cpm_find(pybind11 2.10.1
    GLOBAL_TARGETS pybind11

    CPM_ARGS

    GITHUB_REPOSITORY pybind/pybind11
    GIT_TAG v2.10.1
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)

# https://pybind11.readthedocs.io/en/stable/compiling.html#configuration-variables
#    set(PYBIND11_PYTHON_VERSION 3.6) # It doesn't find python in manylinux2014 image
if (NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE /usr/bin/python3 PARENT_SCOPE)
endif ()
