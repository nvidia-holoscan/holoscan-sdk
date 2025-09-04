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

include(FetchContent)
include(GNUInstallDirs)

find_package(Vulkan REQUIRED)

set(imgui_TAG "f3373780668fba1f9bd64c208d05c20b781c9a39")
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG "${imgui_TAG}"
  GIT_PROGRESS TRUE
  PATCH_COMMAND git apply "${CMAKE_CURRENT_SOURCE_DIR}/imgui.patch" || true
  )

message(STATUS "Fetching imgui@${imgui_TAG}")
FetchContent_MakeAvailable(imgui)

add_library(imgui_backend STATIC)

target_sources(imgui_backend
  PRIVATE
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
  )

target_include_directories(imgui_backend
  PUBLIC
    ${imgui_SOURCE_DIR}
  )

target_compile_definitions(imgui_backend
  PRIVATE
    # prevent the GLFW header from including the OpenGL header
    GLFW_INCLUDE_NONE
  )

target_link_libraries(imgui_backend
  PRIVATE
    Vulkan::Vulkan
    glfw
    holoscan_security_flags
  )

# export the ImGUI installation of Holoviz as a static library
add_library(holoviz_imgui STATIC)
add_library(holoscan::viz::imgui ALIAS holoviz_imgui)

target_sources(holoviz_imgui
  PRIVATE
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
  )

target_include_directories(holoviz_imgui
  PUBLIC
    $<BUILD_INTERFACE:${imgui_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>/holoviz/imgui
  )

set_target_properties(holoviz_imgui
  PROPERTIES POSITION_INDEPENDENT_CODE ON
  EXPORT_NAME viz::imgui
  )

install(
  FILES
    ${imgui_SOURCE_DIR}/imconfig.h
    ${imgui_SOURCE_DIR}/imgui_internal.h
    ${imgui_SOURCE_DIR}/imgui.h
    ${imgui_SOURCE_DIR}/imstb_rectpack.h
    ${imgui_SOURCE_DIR}/imstb_textedit.h
    ${imgui_SOURCE_DIR}/imstb_truetype.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoviz/imgui
  COMPONENT "holoscan-dependencies"
  )

install(
  FILES
    ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoviz/imgui/misc/cpp
  COMPONENT "holoscan-dependencies"
  )