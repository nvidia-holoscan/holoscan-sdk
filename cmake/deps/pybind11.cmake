# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

rapids_cpm_find(pybind11 2.13.6
    GLOBAL_TARGETS pybind11

    CPM_ARGS

    GITHUB_REPOSITORY pybind/pybind11
    GIT_TAG v2.13.6
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)

# https://pybind11.readthedocs.io/en/stable/compiling.html#configuration-variables
#    set(PYBIND11_PYTHON_VERSION 3.6) # It doesn't find python in manylinux2014 image
if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE /usr/bin/python3 PARENT_SCOPE)
endif()
