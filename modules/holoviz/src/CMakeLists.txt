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

set(PROJECT_NAME clara_holoviz)

include(GNUInstallDirs)

find_package(CUDAToolkit REQUIRED)
find_package(X11 REQUIRED)

add_library(${PROJECT_NAME} SHARED)
add_library(clara::holoviz ALIAS ${PROJECT_NAME})

# compile GLSL source files to SPIR-V
include("${nvpro_core_CMAKE_DIR}/utilities.cmake")

set(GLSL_SOURCE_FILES)
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/geometryShader.glsl.frag")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/geometryShader.glsl.vert")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/geometryTextShader.glsl.frag")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/geometryTextShader.glsl.vert")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/imageLUTFloatShader.glsl.frag")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/imageLUTUIntShader.glsl.frag")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/imageShader.glsl.frag")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/imageShader.glsl.vert")

compile_glsl(
    SOURCE_FILES ${GLSL_SOURCE_FILES}
    DST "${CMAKE_CURRENT_BINARY_DIR}/vulkan/spv"
    VULKAN_TARGET "vulkan1.2"
    HEADER ON
    )

set_source_files_properties(${GLSL_SOURCE_FILES} PROPERTIES GENERATED TRUE)

# sources
target_sources(${PROJECT_NAME}
    PRIVATE
        Context.cpp
        Holoviz.cpp
        GLFWWindow.cpp
        ExclusiveWindow.cpp

        cuda/Convert.cu

        layers/GeometryLayer.cpp
        layers/ImageLayer.cpp
        layers/ImGuiLayer.cpp
        layers/Layer.cpp

        vulkan/Vulkan.cpp

        ${GLSL_SOURCE_FILES}
    )

target_include_directories(${PROJECT_NAME}
    PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>

    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${nvpro_core_INT_IMGUI_DIR}>
        $<BUILD_INTERFACE:${nvpro_core_EXT_IMGUI_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        X11::X11
        nvpro_core
        clara::holoviz::imgui

    PUBLIC
        CUDA::cuda_driver
    )

# export only needed symbols
set(EXPORT_MAP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/export.map)
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        LINK_DEPENDS ${EXPORT_MAP_FILE}
)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--version-script=${EXPORT_MAP_FILE}")

# set library version
string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.([0-9]+)$" "\\1" SOVERSION ${CLARA_HOLOVIZ_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${CLARA_HOLOVIZ_VERSION}
    SOVERSION ${SOVERSION})

# install
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Config
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT ${PROJECT_NAME}Config
    DESTINATION cmake
)

install(
    DIRECTORY
        holoviz
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
