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

# Unfortunately, yaml-cpp project's CMakeLists.txt registers the user package
# (see below) which creates an item in '~/.cmake/packages/yaml-cpp/' and makes
# `find_package()` command in CPM try to look at the registered packages for
# 'yaml-cpp'.
# If the user configures CMake build twice consecutively without building
# source, the second configure will use a package in the user package registry
# (specified by '~/.cmake/packages/yaml-cpp/xxxxx' which refers to
# '${CMAKE_BINARY_DIR}/_deps/yaml-cpp-build')
# causing a failure when building the source tree because
# '_deps/yaml-cpp-build/libyaml-cpp.a', needed by'libholoscan-embedded.so' is
# missing.
#
# export(PACKAGE yaml-cpp)
#
# To prevent the situation, we set CMAKE_FIND_USE_PACKAGE_REGISTRY to FALSE
# (https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_USE_PACKAGE_REGISTRY.html#variable:CMAKE_FIND_USE_PACKAGE_REGISTRY)
set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)

# https://github.com/cpm-cmake/CPM.cmake/wiki/More-Snippets#yaml-cpp
set(YAML_CPP_CPM_ARGS
    GITHUB_REPOSITORY jbeder/yaml-cpp
    GIT_TAG yaml-cpp-0.6.3
    OPTIONS
    "YAML_CPP_BUILD_TESTS Off"
    "YAML_CPP_BUILD_CONTRIB Off"
    "YAML_CPP_BUILD_TOOLS Off"
)
rapids_cpm_find(yaml-cpp 0.6.3
    GLOBAL_TARGETS yaml-cpp
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports
    INSTALL_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports

    CPM_ARGS
    ${YAML_CPP_CPM_ARGS}
)

# If CPM is using the local package of yaml-cpp, it doesn't create a custom 'Findyaml-cpp.cmake' in
# '${CMAKE_BINARY_DIR}/CPM_modules' folder.
# We force to create one for yaml-cpp because we will copy
# ${CMAKE_BINARY_DIR}/CPM_modules/Findyaml-cpp.cmake for the installation of 'holoscan-embedded'
# package.
cpm_create_module_file(yaml-cpp "CPMAddPackage(\"${YAML_CPP_CPM_ARGS}\")")

if(yaml-cpp_ADDED)
    # Set PIC to prevent the following error message
    # : relocation R_X86_64_PC32 against symbol `_ZTVN4YAML9ExceptionE' can not be used when making a shared object; recompile with -fPIC
    set_target_properties(yaml-cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
