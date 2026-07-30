# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(PROJECT_NAME holoviz)

include(GNUInstallDirs)
include(GenHeaderFromBinaryFile)

find_package(Vulkan REQUIRED)

add_library(${PROJECT_NAME} SHARED)
add_library(holoscan::viz ALIAS ${PROJECT_NAME})

# compile GLSL source files to SPIR-V
include("${nvpro_core_CMAKE_DIR}/utilities.cmake")

set(GLSL_SOURCE_FILES)
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/geometry_color_shader.glsl.vert")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/geometry_shader.glsl.frag")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/geometry_shader.glsl.vert")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/image_shader.glsl.frag")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/image_shader.glsl.vert")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/imgui_shader.glsl.frag")
list(APPEND GLSL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/imgui_shader.glsl.vert")

set(GLSLANGVALIDATOR ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})
compile_glsl(
    SOURCE_FILES ${GLSL_SOURCE_FILES}
    DST "${CMAKE_CURRENT_BINARY_DIR}/vulkan/spv"
    VULKAN_TARGET "vulkan1.2"
    HEADER ON
    )

set_source_files_properties(${GLSL_SOURCE_FILES} PROPERTIES GENERATED TRUE)

# generate the header file to embed the font
gen_header_from_binary_file(TARGET ${PROJECT_NAME} FILE_PATH "fonts/Roboto-Bold.ttf")

# sources
target_sources(${PROJECT_NAME}
    PRIVATE
        context.cpp
        exclusive_window.cpp
        glfw_window.cpp
        headless_window.cpp
        holoviz.cpp
        window.cpp

        cuda/convert.cu
        cuda/cuda_service.cpp
        cuda/gen_depth_map.cu
        cuda/gen_primitive_vertices.cu

        layers/geometry_layer.cpp
        layers/image_layer.cpp
        layers/im_gui_layer.cpp
        layers/layer.cpp

        vulkan/buffer.cpp
        vulkan/format_util.cpp
        vulkan/framebuffer_sequence.cpp
        vulkan/resource.cpp
        vulkan/texture.cpp
        vulkan/vulkan_app.cpp

        ${GLSL_SOURCE_FILES}
    )

target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        # prevent the GLFW header from including the OpenGL header
        GLFW_INCLUDE_NONE
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
        Vulkan::Vulkan
        glfw
        magic_enum::magic_enum
        nvpro_core
        holoscan::viz::imgui
        holoscan::logger
        holoscan_security_flags

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

# set library name and version
set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME holoscan_viz
    EXPORT_NAME viz
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
)

# install binaries
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Config
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

# install headers
install(
    DIRECTORY
        holoviz
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
