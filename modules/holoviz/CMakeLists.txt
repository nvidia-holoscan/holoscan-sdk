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

cmake_minimum_required(VERSION 3.17)

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/VERSION CLARA_HOLOVIZ_VERSION)

project(ClaraHoloViz
    DESCRIPTION "Clara Holoviz"
    VERSION ${CLARA_HOLOVIZ_VERSION}
    LANGUAGES CXX CUDA
    )

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package")
# avoid 'Up-to-date' install messages
set(CMAKE_INSTALL_MESSAGE LAZY)
# strip release binaries
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(_INSTALL_TARGET "install/strip")
else()
    set(_INSTALL_TARGET "install")
endif()
# show progress when populating external content
set(FETCHCONTENT_QUIET OFF)

# this generates a 'compile_commands.json' file which can be read by VSCode to
#configure the include paths for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# support Maxwell and up
set(CMAKE_CUDA_ARCHITECTURES 50)

add_subdirectory(thirdparty)

if(HOLOSCAN_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

add_subdirectory(src)
if(HOLOSCAN_BUILD_TESTS)
    add_subdirectory(tests)
endif()
