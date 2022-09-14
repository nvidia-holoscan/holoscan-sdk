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

# fetch the dependencies
include(FetchContent)
include(GNUInstallDirs)

FetchContent_Declare(
    nvpro_core
    GIT_REPOSITORY https://github.com/nvpro-samples/nvpro_core.git
    GIT_TAG 6df347cc91535c1a080b467e45de10fe6789c542
    GIT_PROGRESS TRUE
    SOURCE_DIR nvpro_core
)
FetchContent_GetProperties(nvpro_core)
if(NOT nvpro_core_POPULATED)
    FetchContent_Populate(nvpro_core)

    # setup nvpro_core
    set(BASE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)

    _add_project_definitions(${CMAKE_PROJECT_NAME})

    # additions from packages needed for this app
    # add refs  in LIBRARIES_OPTIMIZED
    # add refs  in LIBRARIES_DEBUG
    # add files in PACKAGE_SOURCE_FILES
    _add_package_VulkanSDK()
    _add_package_IMGUI()

    # process the rest of some cmake code that needs to be done *after* the packages add
    _add_nvpro_core_lib()

    # nvpro_core is using add_definitions() and include_directories(), convert these to target properties
    get_directory_property(NVPRO_CORE_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES)
    target_include_directories(nvpro_core
        PUBLIC
            ${NVPRO_CORE_INCLUDE_DIRECTORIES}
        )
    get_directory_property(NVPRO_CORE_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)
    # nvpro_core expects GLFW 3.4 which is not yet release. 3.4 added GLFW_CONTEXT_DEBUG as an alias to GLFW_OPENGL_DEBUG_CONTEXT,
    # we do this manually.
    target_compile_definitions(nvpro_core
        PRIVATE
            -DGLFW_CONTEXT_DEBUG=GLFW_OPENGL_DEBUG_CONTEXT
        PUBLIC
            ${NVPRO_CORE_COMPILE_DEFINITIONS}
        )

    target_sources(nvpro_core
        PRIVATE
            ${COMMON_SOURCE_FILES})

    # export the nvpro_core cmake directory to provide access to e.g. utilities.cmake for compile_glsl_directory()
    set(nvpro_core_CMAKE_DIR "${nvpro_core_SOURCE_DIR}/cmake" CACHE INTERNAL "nvpro_core cmake dir" FORCE)

    set(nvpro_core_INT_IMGUI_DIR "${nvpro_core_SOURCE_DIR}/imgui" CACHE INTERNAL "nvpro_core internal imgui dir" FORCE)
    set(nvpro_core_EXT_IMGUI_DIR "${nvpro_core_SOURCE_DIR}/third_party/imgui" CACHE INTERNAL "nvpro_core external imgui dir" FORCE)

    # export the ImGUI installation of nvpro_core as a static library
    set(PROJECT_NAME clara_holoviz_imgui)
    add_library(${PROJECT_NAME} STATIC)
    add_library(clara::holoviz::imgui ALIAS ${PROJECT_NAME})

    target_sources(${PROJECT_NAME}
        PRIVATE
            ${nvpro_core_EXT_IMGUI_DIR}/imgui.cpp
            ${nvpro_core_EXT_IMGUI_DIR}/imgui_draw.cpp
            ${nvpro_core_EXT_IMGUI_DIR}/imgui_tables.cpp
            ${nvpro_core_EXT_IMGUI_DIR}/imgui_widgets.cpp
        )

    target_include_directories(${PROJECT_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${nvpro_core_EXT_IMGUI_DIR}>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>/holoviz/imgui
        )

    install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

    install(EXPORT ${PROJECT_NAME}Config
        DESTINATION cmake
        )

    install(
        FILES
            ${nvpro_core_EXT_IMGUI_DIR}/imconfig.h
            ${nvpro_core_EXT_IMGUI_DIR}/imgui_internal.h
            ${nvpro_core_EXT_IMGUI_DIR}/imgui.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoviz/imgui
        )

endif()
