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

include(GNUInstallDirs)

find_package(Vulkan REQUIRED)

# nvpro_core lib
# - files have been copied from commit 6df347cc91535c1a080b467e45de10fe6789c542 of nvpro_core repo
add_library(nvpro_core STATIC)

target_sources(nvpro_core
  PRIVATE
    nvh/cameramanipulator.cpp
    nvh/nvprint.cpp

    nvp/perproject_globals.cpp

    nvvk/buffersuballocator_vk.cpp
    nvvk/commands_vk.cpp
    nvvk/context_vk.cpp
    nvvk/debug_util_vk.cpp
    nvvk/descriptorsets_vk.cpp
    nvvk/error_vk.cpp
    nvvk/extensions_vk.cpp
    nvvk/images_vk.cpp
    nvvk/memallocator_dedicated_vk.cpp
    nvvk/memallocator_vk.cpp
    nvvk/memorymanagement_vk.cpp
    nvvk/nsight_aftermath_vk.cpp
    nvvk/pipeline_vk.cpp
    nvvk/resourceallocator_vk.cpp
    nvvk/samplers_vk.cpp
    nvvk/shadermodulemanager_vk.cpp
    nvvk/stagingmemorymanager_vk.cpp
    nvvk/swapchain_vk.cpp
  )

target_compile_definitions(nvpro_core
  PRIVATE
    PROJECT_NAME="Holoviz"
    # nvpro_core expects GLFW 3.4 which is not yet released. 3.4 added GLFW_CONTEXT_DEBUG
    # as an alias to GLFW_OPENGL_DEBUG_CONTEXT, we do this manually.
    -DGLFW_CONTEXT_DEBUG=GLFW_OPENGL_DEBUG_CONTEXT
    NVP_SUPPORTS_VULKANSDK
    # prevent the GLFW header from including the OpenGL header
    GLFW_INCLUDE_NONE
  )

target_include_directories(nvpro_core
  PUBLIC
    ${imgui_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/nvp
  )

target_link_libraries(nvpro_core
  PUBLIC
    imgui_backend
    Vulkan::Vulkan
    glfw
    holoscan_security_flags
  )

set_target_properties(nvpro_core
  PROPERTIES POSITION_INDEPENDENT_CODE ON
  )

# export the nvpro_core cmake directory to provide access to e.g. utilities.cmake for compile_glsl_directory()
set(nvpro_core_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake" CACHE INTERNAL "nvpro_core cmake dir" FORCE)
