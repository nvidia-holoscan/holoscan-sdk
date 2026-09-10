# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" HOLOVIZ_PROJECT_VERSION)

project(holoscan_visualization
    VERSION ${HOLOVIZ_PROJECT_VERSION}
    DESCRIPTION "Holoscan Visualization module"
    LANGUAGES CXX
)

include(GenerateExportHeader)
include(GNUInstallDirs)

# Holoviz links statically into the Holoscan Visualization module, tests, and examples by default.
# The shared build supports a module-owned runtime library and is deliberately independent of
# the global BUILD_SHARED_LIBS, so an SDK build with shared core still absorbs holoviz statically.
option(HOLOVIZ_BUILD_SHARED "Build HoloViz as a shared library" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# A top-level Holoscan Visualization build is against an installed SDK. The backend is otherwise
# built with the SDK and is not supported as an independent project.
if(holoscan_visualization_IS_TOP_LEVEL)
    set(HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT holoscan-visualization)
    if(NOT DEFINED HOLOSCAN_BUILD_EXAMPLES)
        option(HOLOSCAN_BUILD_EXAMPLES "Build HoloViz examples" OFF)
    endif()
    if(NOT DEFINED HOLOSCAN_BUILD_TESTS)
        option(HOLOSCAN_BUILD_TESTS "Build HoloViz tests" ON)
    endif()
    find_package(holoscan CONFIG QUIET)
    if(NOT holoscan_FOUND)
        message(FATAL_ERROR
            "Top-level Holoscan Visualization builds require an installed Holoscan SDK. Set holoscan_ROOT "
            "to its install prefix or holoscan_DIR to its package-config directory."
        )
    endif()

    # RAPIDS CMake is build infrastructure rather than part of the installed Holoscan package.
    # Initialize it from the surrounding SDK source tree so every dependency uses the same
    # central SDK recipe in SDK and module builds.
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/modules/SetupRapidsCMake.cmake")
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/setup_dependencies.cmake")
else()
    set(HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT holoscan-core)
endif()

include(holoviz_dependencies)

if(HOLOVIZ_BUILD_SHARED)
    add_library(holoviz SHARED)
else()
    add_library(holoviz STATIC)
endif()
add_library(holoviz::holoviz ALIAS holoviz)
holoviz_configure_runtime_search_path(holoviz)

# Everything is hidden unless a declaration opts in with HOLOVIZ_EXPORT, so the DSO exports
# only the module's own API and never a dependency's symbols.
generate_export_header(holoviz
    BASE_NAME HOLOVIZ
    EXPORT_MACRO_NAME HOLOVIZ_EXPORT
    NO_EXPORT_MACRO_NAME HOLOVIZ_NO_EXPORT
    DEPRECATED_MACRO_NAME HOLOVIZ_DEPRECATED
    STATIC_DEFINE HOLOVIZ_STATIC_DEFINE
    INCLUDE_GUARD_NAME HOLOVIZ_EXPORT_H_GENERATED
    EXPORT_FILE_NAME "${holoscan_visualization_BINARY_DIR}/include/holoviz/holoviz_export.h"
)

target_include_directories(holoviz
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${holoscan_visualization_BINARY_DIR}/include>"
)

# A static build has no import/export attributes to apply; the generated header keys off this.
target_compile_definitions(holoviz
    PUBLIC
        $<$<NOT:$<BOOL:${HOLOVIZ_BUILD_SHARED}>>:HOLOVIZ_STATIC_DEFINE>
        $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
        $<$<PLATFORM_ID:Windows>:WIN32_LEAN_AND_MEAN>
)

target_compile_features(holoviz PUBLIC cxx_std_20)

# MSVC defaults to its traditional (non-conformant) preprocessor, which lacks C++20 __VA_OPT__
# as used by the VIZ_VK_DEVICE_FUNC macros in the Vulkan RHI.
target_compile_options(holoviz PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:preprocessor>)

# Bundled Dear ImGui headers for the text module's debug UI-callback mode. They are reachable
# only through <holoviz/text/imgui_ui.hpp>; the rest of the text API forward-declares ImGui.
# Copy them into the generated include tree so every build-tree consumer resolves the exact
# headers this library was compiled with.
get_property(_holoviz_imgui_source_dir GLOBAL PROPERTY HOLOVIZ_IMGUI_SOURCE_DIR)
if(NOT _holoviz_imgui_source_dir)
    message(FATAL_ERROR
        "Holoscan Visualization: the imgui dependency did not record HOLOVIZ_IMGUI_SOURCE_DIR")
endif()
file(COPY
    "${_holoviz_imgui_source_dir}/imgui.h"
    "${_holoviz_imgui_source_dir}/imconfig.h"
    DESTINATION "${holoscan_visualization_BINARY_DIR}/include/holoviz/imgui"
)
unset(_holoviz_imgui_source_dir)

# Top-level module packages ship the canonical SDK license and their notice at the package root.
# When built as part of the Holoscan SDK, the SDK installs these files centrally.
if(holoscan_visualization_IS_TOP_LEVEL)
    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/../../LICENSE.txt"
        NOTICE.txt
        DESTINATION .
        COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    )
endif()

add_subdirectory(src)

# HolovizOp is the Holoscan-facing part of this module. Keep its implementation and installation
# rules beside the backend it consumes so disabling HoloViz removes the complete feature.
add_subdirectory(operators)

if(HOLOSCAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NOT holoscan_visualization_IS_TOP_LEVEL)
    set(HOLOSCAN_INSTALL_TARGETS "${HOLOSCAN_INSTALL_TARGETS}" PARENT_SCOPE)
endif()

if(HOLOSCAN_BUILD_TESTS)
    include(CTest)
    enable_testing()
    if(BUILD_TESTING)
        function(holoviz_add_dependency_dispatch_test build_mode slang_mode)
            set(_holoviz_dependency_test_name
                "CMAKE_HOLOVIZ_DEPENDENCIES.${build_mode}.${slang_mode}")
            set(_holoviz_dependency_test_args
                -DHOLOVIZ_CMAKE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/cmake
                -DHOLOSCAN_CMAKE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../cmake
                -DHOLOVIZ_TEST_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/cmake-tests/cache-${build_mode}-${slang_mode}
                -DHOLOVIZ_TEST_MODE=${build_mode}
                -DHOLOVIZ_TEST_SLANG_MODE=${slang_mode}
            )
            add_test(NAME ${_holoviz_dependency_test_name}
                COMMAND ${CMAKE_COMMAND}
                    -S "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/dependency_dispatch"
                    -B "${CMAKE_CURRENT_BINARY_DIR}/cmake-tests/dependency-${build_mode}-${slang_mode}"
                    ${_holoviz_dependency_test_args}
            )
            unset(_holoviz_dependency_test_args)
            unset(_holoviz_dependency_test_name)
        endfunction()

        holoviz_add_dependency_dispatch_test(embedded source)
        holoviz_add_dependency_dispatch_test(holoscan_module source)
        holoviz_add_dependency_dispatch_test(holoscan_module auto)
        if((CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Windows") AND
            CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64|x86_64|AMD64)$")
            holoviz_add_dependency_dispatch_test(holoscan_module prebuilt)
        endif()
        holoviz_add_dependency_dispatch_test(holoscan_module external_root)
        holoviz_add_dependency_dispatch_test(holoscan_module external_dir)
        holoviz_add_dependency_dispatch_test(holoscan_module external_system)

        add_subdirectory(tests)
    endif()
endif()
