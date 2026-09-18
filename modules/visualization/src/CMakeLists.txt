# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# include(GenHeaderFromBinaryFile)

# HoloViz is a shared library regardless of the global BUILD_SHARED_LIBS: HolovizOp links it, and
# external visualization operators consume it as holoscan::visualization::holoviz through the
# `visualization` component of the installed holoscan package. A single shared build keeps both
# consumers on the same runtime library. The alias gives the build tree the same namespaced name
# the installed package exports, so tests and examples link one name in both contexts.
add_library(holoviz SHARED)
add_library(holoscan::visualization::holoviz ALIAS holoviz)
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
        "$<BUILD_INTERFACE:${holoscan_visualization_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${holoscan_visualization_BINARY_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_compile_definitions(holoviz
    PUBLIC
        $<$<PLATFORM_ID:Windows>:VK_USE_PLATFORM_WIN32_KHR>
        $<$<PLATFORM_ID:Windows>:NOMINMAX>
        $<$<PLATFORM_ID:Windows>:WIN32_LEAN_AND_MEAN>
)

target_compile_features(holoviz PUBLIC cxx_std_20)

# MSVC defaults to its traditional (non-conformant) preprocessor, which lacks C++20 __VA_OPT__
# as used by the VIZ_VK_DEVICE_FUNC macros in the Vulkan backend.
target_compile_options(holoviz PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:preprocessor>)

# Bundled Dear ImGui headers for the imgui module's debug UI-callback mode. They are reachable
# only through <holoviz/imgui/ui.hpp>; the rest of the imgui and text APIs forward-declare ImGui.
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
    DESTINATION "${holoscan_visualization_BINARY_DIR}/include/holoviz/imgui/vendor"
)
unset(_holoviz_imgui_source_dir)

# sources
target_sources(holoviz
    PRIVATE
        logger.cpp

        adapter.cpp
        buffer.cpp
        command_list.cpp
        descriptor.cpp
        device.cpp
        display.cpp
        external_semaphore.cpp
        image.cpp
        pipeline.cpp
        query.cpp
        render_pass.cpp
        surface.cpp
        swapchain.cpp
        window.cpp
        slang_compiler.cpp

        vulkan/vk_adapter.cpp
        vulkan/vk_impls.cpp
        vulkan/vk_buffer.cpp
        vulkan/vk_command_list.cpp
        vulkan/vk_external_semaphore.cpp
        vulkan/vk_device.cpp
        vulkan/vk_display.cpp
        vulkan/vk_surface.cpp
        vulkan/vk_swapchain.cpp
        vulkan/vk_pipeline.cpp
        vulkan/vk_image.cpp
        vulkan/vk_errors.cpp
        vulkan/vk_enum_conversions.cpp
        vulkan/vk_descriptor.cpp
        vulkan/vk_query.cpp
        vulkan/vk_render_pass.cpp

        window_systems/glfw_window.cpp

        cuda/cu_errors.cpp
        cuda/cu_device.cpp
        cuda/cu_enum_conversions.cpp

        imgui/detail/glfw_keymap.cpp
        imgui/renderer.cpp
        imgui/shader.cpp
        text/text_renderer.cpp
        text/monospace_shader.cpp
        text/monospace_text_renderer.cpp
    )

# Direct-display ownership classification + acquisition is the one genuinely
# platform-specific corner of HoloViz: WinRT (Windows.Devices.Display.Core +
# VK_NV_acquire_winrt_display) on Windows, libdrm (VK_EXT_acquire_drm_display)
# on Linux. See src/display_systems/display_ownership.hpp.
if(WIN32)
    target_sources(holoviz PRIVATE display_systems/display_ownership_win32.cpp)
    # C++/WinRT projection over the OS WinRT APIs (headers ship with the Windows SDK).
    target_link_libraries(holoviz PRIVATE windowsapp.lib)
elseif(UNIX)
    target_sources(holoviz PRIVATE display_systems/display_ownership_drm.cpp)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(Libdrm REQUIRED IMPORTED_TARGET libdrm)
    target_link_libraries(holoviz PRIVATE PkgConfig::Libdrm)
    # X11/RandR direct-display (VK_EXT_acquire_xlib_display) is optional: headless
    # and embedded builds without the X dev packages keep the DRM-only path.
    # VK_USE_PLATFORM_XLIB_XRANDR_EXT must be target-wide (volk only declares and
    # loads vkAcquireXlibDisplayEXT/vkGetRandROutputDisplayEXT under it, and
    # VOLK_IMPLEMENTATION lives in vk_impls.cpp); it stays PRIVATE because no
    # public header exposes the extension.
    pkg_check_modules(X11Xrandr IMPORTED_TARGET x11 xrandr)
    if(X11Xrandr_FOUND)
        target_sources(holoviz PRIVATE display_systems/display_ownership_x11.cpp)
        target_link_libraries(holoviz PRIVATE PkgConfig::X11Xrandr)
        target_compile_definitions(holoviz PRIVATE
            VK_USE_PLATFORM_XLIB_XRANDR_EXT
            HOLOVIZ_HAS_XLIB_XRANDR
        )
    endif()
endif()

# Dear ImGui is compiled directly into holoviz instead of a static archive: object files are not
# subject to --exclude-libs, so with IMGUI_API pinned to default visibility the ImGui entry
# points stay exported for the debug UI-callback mode (<holoviz/imgui/ui.hpp>) while every
# other dependency stays hidden.

get_property(_holoviz_imgui_source_dir GLOBAL PROPERTY HOLOVIZ_IMGUI_SOURCE_DIR)
target_sources(holoviz
    PRIVATE
        "${_holoviz_imgui_source_dir}/imgui.cpp"
        "${_holoviz_imgui_source_dir}/imgui_draw.cpp"
        "${_holoviz_imgui_source_dir}/imgui_tables.cpp"
        "${_holoviz_imgui_source_dir}/imgui_widgets.cpp"
    )
target_include_directories(holoviz
    PRIVATE
        "${_holoviz_imgui_source_dir}"
    )
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_definitions(holoviz PRIVATE
        "IMGUI_API=__attribute__((visibility(\"default\")))")
elseif(MSVC)
    target_compile_definitions(holoviz PRIVATE "IMGUI_API=__declspec(dllexport)")
endif()
unset(_holoviz_imgui_source_dir)

target_compile_definitions(holoviz
    PRIVATE
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        GLFW_INCLUDE_NONE
    )

# Internal headers (detail/*_impl.hpp) are included as "detail/..." from
# anywhere under src/; they are deliberately not installed.
target_include_directories(holoviz
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

# HoloViz owns its error and logging vocabulary (<holoviz/errors.hpp>, <holoviz/expected.hpp>,
# <holoviz/logger.hpp>) and does not link Holoscan. tl-expected is the one public dependency,
# because holoviz::expected is an alias for tl::expected in the headers. Integrations that want
# HoloViz logs in the Holoscan logger include <holoviz/integrations/holoscan_logger.hpp>, which is
# header-only and links holoscan::logger on the consumer's side.
target_link_libraries(holoviz
    PUBLIC
        CUDA::cuda_driver
    PRIVATE
        magic_enum::magic_enum
        # Pinned Khronos headers only — no loader link; volk dlopens libvulkan at runtime.
        Vulkan::Headers
        Vulkan::UtilityHeaders
        volk::volk_headers
        GPUOpen::VulkanMemoryAllocator
    )

# GLFW is a static archive absorbed into libholoviz (see cmake/deps/glfw.cmake), so a PRIVATE
# link propagates nothing and is not recorded in the exported target.
target_link_libraries(holoviz PRIVATE glfw)

# Slang is an imported prebuilt shared library deployed beside HoloViz (or by its external
# supplier) and is not exported. A PRIVATE shared dependency would still be recorded as
# IMPORTED_LINK_DEPENDENT_LIBRARIES on the exported holoviz target, which
# tests/system/holoviz_installed_consumer rejects; keeping it in the build interface only leaves
# the installed target free of it. Installed consumers do not need the target.
target_link_libraries(holoviz PUBLIC $<BUILD_INTERFACE:slang::slang>)

target_link_libraries(holoviz PUBLIC tl::expected)

# The SDK's hardening flags are an internal INTERFACE target defined by
# cmake/modules/BuildSecurityFlags.cmake and are not part of the installed Holoscan package,
# so they are only available in the SDK build. A top-level module build gets whatever hardening
# the toolchain and CMAKE_<LANG>_FLAGS provide; it is not the shipping configuration.
if(TARGET holoscan_security_flags)
    target_link_libraries(holoviz PRIVATE holoscan_security_flags)
endif()

set_target_properties(holoviz PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    OUTPUT_NAME holoviz
    EXPORT_NAME visualization::holoviz
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    VISIBILITY_INLINES_HIDDEN YES
)

# Static dependencies (GLFW, volk, VMA) must not re-export their symbols through this DSO;
# without this the linker would surface every archive symbol it pulled in. In the SDK build the
# shared helper also adds the SDK's --no-undefined / --gc-sections hardening.
if(COMMAND holoscan_configure_linux_shared_link)
    holoscan_configure_linux_shared_link(holoviz)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(holoviz PRIVATE
        LINKER:--exclude-libs,ALL
    )
endif()

# HoloViz is exported through the module's holoscan-visualization-exports set as
# holoscan::visualization::holoviz (EXPORT_NAME above, holoscan:: namespace). HolovizOp links it
# privately as a shared library, so it must live in the same export set; find_package(holoscan
# COMPONENTS visualization) and find_package(holoscan-visualization) import both.
#
# In install(TARGETS), COMPONENT belongs to the artifact group it follows, so each group names the
# module component explicitly. A single trailing COMPONENT would cover RUNTIME only and let the
# Linux shared library fall back to CMAKE_INSTALL_DEFAULT_COMPONENT_NAME, shipping the export
# metadata without libholoviz in a component-wise install or package.
install(TARGETS holoviz
    EXPORT holoscan-visualization-exports
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
)

# The module's own headers (<holoviz/**>, <holoviz/text/**>, errors/expected/logger and the
# generated export header) ship with the module so authors of custom operators can build against
# HoloViz. HolovizOp's installed API no longer re-exports them: it installs <holoscan/visualization/*>
# only. The bundled Dear ImGui headers go with them because <holoviz/imgui/ui.hpp> includes
# them.
install(
    DIRECTORY "${holoscan_visualization_SOURCE_DIR}/include/holoviz"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.hpp"
)
install(FILES "${holoscan_visualization_BINARY_DIR}/include/holoviz/holoviz_export.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoviz
    COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
)
install(
    DIRECTORY "${holoscan_visualization_BINARY_DIR}/include/holoviz/imgui/vendor"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoviz/imgui
    COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    FILES_MATCHING PATTERN "*.h"
)
