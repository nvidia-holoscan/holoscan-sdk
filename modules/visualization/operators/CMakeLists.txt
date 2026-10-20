# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GenerateExportHeader)

# The RenderBackend/PresentBackend seam and its HoloViz implementations are compiled once into an
# object library that both the operator DSO and the operator unit tests consume. They consume
# the layer/sink declarations from the public operator header, which builds on Holoscan Core.
# Tests link the object library rather than the DSO because the seam headers are private to the
# operator and never installed or exported.
add_library(op_holoviz_backends OBJECT
    src/common/holoviz_conversions.cpp
    src/holoviz_op/backends/backend_tokens.cpp
    src/holoviz_op/backends/cuda_present_backend.cpp
    src/holoviz_op/backends/display_surface.cpp
    src/holoviz_op/backends/holoviz_geometry_layer.cpp
    src/holoviz_op/backends/holoviz_image_layer.cpp
    src/holoviz_op/backends/holoviz_lut_layer.cpp
    src/holoviz_op/backends/holoviz_render_backend.cpp
    src/holoviz_op/backends/holoviz_text_layer.cpp
    src/holoviz_op/backends/rgb_to_rgba.cu
    src/holoviz_op/backends/swap_red_blue.cu
    src/holoviz_op/backends/swapchain_present_backend.cpp
    src/holoviz_op/backends/window_surface.cpp
)

# The layer shaders are Slang sources compiled to SPIR-V at build time (see
# cmake/HolovizShaderHeader.cmake); each generated header exposes a make_shader_blob() factory per
# entry-point pair. The custom commands live in this directory beside the object library, so
# attaching the headers as sources orders generation before the layer object compiles. Each shader
# shares its push-constant struct with its layer through a *_interface.hpp beside it, opened in the
# same per-shader NAMESPACE as the generated header so the three PushConstants are distinct C++
# types; PUSH_CONSTANTS_TYPE makes the generated header assert that struct against the reflection.
include(HolovizShaderHeader)
set(_op_holoviz_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated")
holoviz_generate_shader_header(
    OUTPUT "${_op_holoviz_generated_dir}/backends/holoviz_geometry.slang.hpp"
    SLANG src/holoviz_op/backends/holoviz_geometry.slang
    NAMESPACE holoscan::visualization::backends::detail::holoviz_geometry
    PAIRS
        line_list=vertex_line_list:fragment_main
        line_strip=vertex_line_strip:fragment_main
        cross=vertex_cross:fragment_main
        rect=vertex_rect:fragment_main
        oval=vertex_oval:fragment_main
    PUSH_CONSTANTS_TYPE holoscan::visualization::backends::detail::holoviz_geometry::PushConstants
    PUSH_CONSTANTS_INCLUDE holoviz_op/backends/holoviz_geometry_interface.hpp
)
holoviz_generate_shader_header(
    OUTPUT "${_op_holoviz_generated_dir}/backends/holoviz_image.slang.hpp"
    SLANG src/holoviz_op/backends/holoviz_image.slang
    NAMESPACE holoscan::visualization::backends::detail::holoviz_image
    PAIRS main=vertex_main:fragment_main
    PUSH_CONSTANTS_TYPE holoscan::visualization::backends::detail::holoviz_image::PushConstants
    PUSH_CONSTANTS_INCLUDE holoviz_op/backends/holoviz_image_interface.hpp
)
holoviz_generate_shader_header(
    OUTPUT "${_op_holoviz_generated_dir}/backends/holoviz_lut.slang.hpp"
    SLANG src/holoviz_op/backends/holoviz_lut.slang
    NAMESPACE holoscan::visualization::backends::detail::holoviz_lut
    PAIRS main=vertex_main:fragment_main
    PUSH_CONSTANTS_TYPE holoscan::visualization::backends::detail::holoviz_lut::PushConstants
    PUSH_CONSTANTS_INCLUDE holoviz_op/backends/holoviz_lut_interface.hpp
)
target_sources(op_holoviz_backends PRIVATE
    "${_op_holoviz_generated_dir}/backends/holoviz_geometry.slang.hpp"
    "${_op_holoviz_generated_dir}/backends/holoviz_image.slang.hpp"
    "${_op_holoviz_generated_dir}/backends/holoviz_lut.slang.hpp"
)
target_include_directories(op_holoviz_backends PRIVATE "${_op_holoviz_generated_dir}")
unset(_op_holoviz_generated_dir)

include(GNUInstallDirs)
# The object library is never installed, so every root is build-interface only. The seam tests
# reach the operator-private ("holoviz_op/...") and module-private ("detail/...", "cuda/...")
# headers through these usage requirements; neither tree is installed.
target_include_directories(op_holoviz_backends PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
    "$<BUILD_INTERFACE:${holoscan_visualization_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${holoscan_visualization_BINARY_DIR}/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
    "$<BUILD_INTERFACE:${holoscan_visualization_SOURCE_DIR}/src>"
)

target_link_libraries(op_holoviz_backends
    # sensor_io is a header-only consequence here, not a real dependency of the backends: the
    # public holoviz_op.hpp (transitively included via types.hpp) declares Input<schema::ImageT>
    # accessors, so anything including it needs sensor_io's include path to compile, even though
    # no backend source touches schema::ImageT directly.
    PUBLIC
      holoscan::core
      holoscan::sensor_io
      holoscan::visualization::holoviz
      CUDA::cuda_driver
      CUDA::cudart
    PRIVATE magic_enum::magic_enum
)

target_compile_features(op_holoviz_backends PUBLIC cxx_std_20)
set_target_properties(op_holoviz_backends PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN YES
)

# HolovizOp: the Holoscan operator over HoloViz. It is exported beside
# holoscan::visualization::holoviz in the module's holoscan-visualization-exports set;
# find_package(holoscan COMPONENTS visualization) and find_package(holoscan-visualization) import both.
# The operator bodies are compiled once for the DSO and for the unit tests that exercise its hidden
# internals. The error bridge and exclusive-display resolver form their own object library because
# one test needs only them. Both use the DSO's export definition, visibility, and include roots.
add_library(op_holoviz_display_objects OBJECT
    src/holoviz_op/error_translation.cpp
    src/holoviz_op/exclusive_display.cpp
)
add_library(op_holoviz_operator_objects OBJECT
    src/holoviz_op/holoviz_op.cpp
    src/holoviz_op/holoviz_op_impl.cpp
)
foreach(_op_holoviz_objects IN ITEMS op_holoviz_display_objects op_holoviz_operator_objects)
    target_link_libraries(${_op_holoviz_objects}
        PUBLIC op_holoviz_backends
        PRIVATE fmt::fmt-header-only magic_enum::magic_enum
    )
    target_compile_definitions(${_op_holoviz_objects} PRIVATE op_holoviz_EXPORTS)
    set_target_properties(${_op_holoviz_objects} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VISIBILITY_INLINES_HIDDEN YES
    )
endforeach()
unset(_op_holoviz_objects)

add_library(op_holoviz SHARED
    src/operator_catalog.cpp
    $<TARGET_OBJECTS:op_holoviz_display_objects>
    $<TARGET_OBJECTS:op_holoviz_operator_objects>
    $<TARGET_OBJECTS:op_holoviz_backends>
)
add_library(holoscan::visualization::op_holoviz ALIAS op_holoviz)

target_include_directories(op_holoviz
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        # HoloViz headers are an implementation detail; nothing under include/holoscan includes them.
        "${holoscan_visualization_SOURCE_DIR}/include"
        "${holoscan_visualization_BINARY_DIR}/include"
        # The operator bodies live under src/<name>op/ and include their private headers and
        # the backend seam as "holoviz_op/...".
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# HoloViz is PRIVATE: the installed <holoscan/visualization/**> headers include no <holoviz/**>
# header (PresentMode is the operator's own enum, translated in src/common/holoviz_conversions.cpp),
# so HolovizOp consumers need only holoscan::core. HoloViz ships beside it in the same export set
# for operators built directly on holoscan::visualization::holoviz.
target_link_libraries(op_holoviz
    # sensor_io is PUBLIC: the installed holoviz_op.hpp declares public image_layer_input()/
    # tensor_layer_input() accessors (needed for graph wiring, e.g.
    # graph.add_flow(producer, viz->image_layer_input(0))) that return Input<schema::ImageT>&,
    # which requires schema::ImageT to be a complete type in this public header regardless of
    # where the port objects themselves live. See the ImageT port alignment design doc's
    # Decision 2 addendum (Cristiana, 2026-09-14, and the second correction after this surfaced
    # a broken example build).
    PUBLIC holoscan::core holoscan::sensor_io
    PRIVATE
        holoscan::visualization::holoviz
        CUDA::cuda_driver
        CUDA::cudart
        fmt::fmt-header-only
        magic_enum::magic_enum
)

target_compile_features(op_holoviz PUBLIC cxx_std_20)
generate_export_header(op_holoviz
    BASE_NAME HOLOSCAN_OPERATOR_HOLOVIZ
    EXPORT_MACRO_NAME HOLOSCAN_OPERATOR_HOLOVIZ_EXPORT
    NO_EXPORT_MACRO_NAME HOLOSCAN_OPERATOR_HOLOVIZ_NO_EXPORT
    INCLUDE_GUARD_NAME HOLOSCAN_OPERATOR_HOLOVIZ_EXPORT_H_GENERATED
    EXPORT_FILE_NAME
        "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/visualization/holoscan_operator_holoviz_export.h"
)
set_target_properties(op_holoviz PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    EXPORT_NAME visualization::op_holoviz
    OUTPUT_NAME holoscan_op_holoviz
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    VISIBILITY_INLINES_HIDDEN YES
)
# SDK-internal helper (cmake/modules/HoloscanTargetConfiguration.cmake); a top-level module build
# without the SDK source tree on CMAKE_MODULE_PATH gets the toolchain defaults instead.
if(COMMAND holoscan_configure_linux_shared_link)
    holoscan_configure_linux_shared_link(op_holoviz)
endif()

# COMPONENT is per artifact group in install(TARGETS); see src/CMakeLists.txt for the rationale.
install(TARGETS op_holoviz
    EXPORT holoscan-visualization-exports
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
)

install(
    DIRECTORY include/holoscan
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.hpp"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/visualization/holoscan_operator_holoviz_export.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/visualization
    COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
)

# The installed operator API is <holoscan/visualization/*> and includes no <holoviz/**> header.
# The module's own headers are installed by the module itself (src/CMakeLists.txt) for operators
# built directly on holoscan::visualization::holoviz; the layer renderer is compiled into the
# operator DSO and its private holoviz_op/ headers are not installed.
