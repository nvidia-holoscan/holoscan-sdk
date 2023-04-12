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

cmake_minimum_required(VERSION 3.22)

# Reset CMAKE_INSTALL_DEFAULT_COMPONENT_NAME to 'holoscan-modules' so that installation caused
# by modules in this folder will not install into the default component that Holoscan SDK
# uses.
# (CMAKE_INSTALL_DEFAULT_COMPONENT_NAME is set to 'holoscan-core' by the root CMakeLists.txt
# before this folder is processed.)
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "holoscan-modules")

include(FetchContent)

FetchContent_Declare(
  holoviz
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/holoviz
)

FetchContent_Declare(
  holoinfer
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/holoinfer
)

FetchContent_MakeAvailable(holoviz holoinfer)