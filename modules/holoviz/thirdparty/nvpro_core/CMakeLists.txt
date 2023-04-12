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

# fetch the dependencies
include(FetchContent)
include(GNUInstallDirs)

FetchContent_Declare(
  nvpro_core
  GIT_REPOSITORY https://github.com/nvpro-samples/nvpro_core.git
  GIT_TAG 6df347cc91535c1a080b467e45de10fe6789c542
  GIT_PROGRESS TRUE
  SOURCE_DIR nvpro_core
  PATCH_COMMAND git apply "${CMAKE_CURRENT_SOURCE_DIR}/nvpro_core.patch" || true
  )

FetchContent_GetProperties(nvpro_core)

if(NOT nvpro_core_POPULATED)
  FetchContent_Populate(nvpro_core)

  # Create the nvpro_core project
  #  We do this manually instead of using the nvpro_core CMakeFile.txt since nvpro_core is
  #  - not using the target_??? commands, this leads to exported private dependencies
  #  - setting multiple global CMAKE_??? variables which overwrite our project variables
  set(nvpro_core_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/nvpro_core)

  find_package(Vulkan REQUIRED)

  add_library(nvpro_core STATIC)

  target_sources(nvpro_core
    PRIVATE
      ${nvpro_core_SOURCE_DIR}/nvh/cameramanipulator.cpp
      ${nvpro_core_SOURCE_DIR}/nvh/nvprint.cpp
      ${nvpro_core_SOURCE_DIR}/nvp/perproject_globals.cpp
    )

  set(nvvk_SOURCE_DIR ${nvpro_core_SOURCE_DIR}/nvvk)

  target_sources(nvpro_core
    PRIVATE
      ${nvvk_SOURCE_DIR}/buffersuballocator_vk.cpp
      ${nvvk_SOURCE_DIR}/commands_vk.cpp
      ${nvvk_SOURCE_DIR}/context_vk.cpp
      ${nvvk_SOURCE_DIR}/debug_util_vk.cpp
      ${nvvk_SOURCE_DIR}/descriptorsets_vk.cpp
      ${nvvk_SOURCE_DIR}/error_vk.cpp
      ${nvvk_SOURCE_DIR}/extensions_vk.cpp
      ${nvvk_SOURCE_DIR}/images_vk.cpp
      ${nvvk_SOURCE_DIR}/memallocator_dedicated_vk.cpp
      ${nvvk_SOURCE_DIR}/memallocator_vk.cpp
      ${nvvk_SOURCE_DIR}/memorymanagement_vk.cpp
      ${nvvk_SOURCE_DIR}/nsight_aftermath_vk.cpp
      ${nvvk_SOURCE_DIR}/pipeline_vk.cpp
      ${nvvk_SOURCE_DIR}/resourceallocator_vk.cpp
      ${nvvk_SOURCE_DIR}/samplers_vk.cpp
      ${nvvk_SOURCE_DIR}/shadermodulemanager_vk.cpp
      ${nvvk_SOURCE_DIR}/stagingmemorymanager_vk.cpp
      ${nvvk_SOURCE_DIR}/swapchain_vk.cpp
    )

  set(imgui_SOURCE_DIR ${nvpro_core_SOURCE_DIR}/third_party/imgui)

  target_sources(nvpro_core
    PRIVATE
      ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
      ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
  )

  target_compile_definitions(nvpro_core
    PRIVATE
      PROJECT_NAME="Holoviz"
      # nvpro_core expects GLFW 3.4 which is not yet released. 3.4 added GLFW_CONTEXT_DEBUG
      # as an alias to GLFW_OPENGL_DEBUG_CONTEXT, we do this manually.
      -DGLFW_CONTEXT_DEBUG=GLFW_OPENGL_DEBUG_CONTEXT
      NVP_SUPPORTS_VULKANSDK
    )

  target_include_directories(nvpro_core
    PUBLIC
      ${imgui_SOURCE_DIR}
      ${nvpro_core_SOURCE_DIR}
      ${nvpro_core_SOURCE_DIR}/nvp
    )

  target_link_libraries(nvpro_core
    PRIVATE
      Vulkan::Vulkan
      glfw
    )

  set_target_properties(nvpro_core
    PROPERTIES POSITION_INDEPENDENT_CODE ON
    )

  # export the nvpro_core cmake directory to provide access to e.g. utilities.cmake for compile_glsl_directory()
  set(nvpro_core_CMAKE_DIR "${nvpro_core_SOURCE_DIR}/cmake" CACHE INTERNAL "nvpro_core cmake dir" FORCE)

  set(nvpro_core_INT_IMGUI_DIR "${nvpro_core_SOURCE_DIR}/imgui"
      CACHE INTERNAL "nvpro_core internal imgui dir" FORCE)
  set(nvpro_core_EXT_IMGUI_DIR "${nvpro_core_SOURCE_DIR}/third_party/imgui"
      CACHE INTERNAL "nvpro_core external imgui dir" FORCE)

  # export the ImGUI installation of nvpro_core as a static library
  add_library(holoviz_imgui STATIC)
  add_library(holoscan::viz::imgui ALIAS holoviz_imgui)

  target_sources(holoviz_imgui
    PRIVATE
      ${nvpro_core_EXT_IMGUI_DIR}/imgui.cpp
      ${nvpro_core_EXT_IMGUI_DIR}/imgui_draw.cpp
      ${nvpro_core_EXT_IMGUI_DIR}/imgui_tables.cpp
      ${nvpro_core_EXT_IMGUI_DIR}/imgui_widgets.cpp
      ${nvpro_core_EXT_IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp
    )

  target_include_directories(holoviz_imgui
    PUBLIC
      $<BUILD_INTERFACE:${nvpro_core_EXT_IMGUI_DIR}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>/holoviz/imgui
    )

  set_target_properties(holoviz_imgui
    PROPERTIES POSITION_INDEPENDENT_CODE ON
    )

  install(TARGETS holoviz_imgui EXPORT holoviz_imguiConfig
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT "holoscan-dependencies"
    )

  install(
    FILES
      ${nvpro_core_EXT_IMGUI_DIR}/imconfig.h
      ${nvpro_core_EXT_IMGUI_DIR}/imgui_internal.h
      ${nvpro_core_EXT_IMGUI_DIR}/imgui.h
      ${nvpro_core_EXT_IMGUI_DIR}/misc/cpp/imgui_stdlib.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoviz/imgui
    COMPONENT "holoscan-dependencies"
    )

endif()
