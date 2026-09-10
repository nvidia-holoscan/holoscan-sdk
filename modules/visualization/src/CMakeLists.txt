# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# include(GenHeaderFromBinaryFile)

# sources
target_sources(holoviz
    PRIVATE
        logger.cpp

        rhi/adapter.cpp
        rhi/buffer.cpp
        rhi/command_list.cpp
        rhi/descriptor.cpp
        rhi/device.cpp
        rhi/display.cpp
        rhi/external_semaphore.cpp
        rhi/image.cpp
        rhi/pipeline.cpp
        rhi/query.cpp
        rhi/render_pass.cpp
        rhi/surface.cpp
        rhi/swapchain.cpp
        rhi/window.cpp
        rhi/slang_compiler.cpp

        rhi/vulkan/vk_adapter.cpp
        rhi/vulkan/vk_impls.cpp
        rhi/vulkan/vk_buffer.cpp
        rhi/vulkan/vk_command_list.cpp
        rhi/vulkan/vk_external_semaphore.cpp
        rhi/vulkan/vk_device.cpp
        rhi/vulkan/vk_display.cpp
        rhi/vulkan/vk_surface.cpp
        rhi/vulkan/vk_swapchain.cpp
        rhi/vulkan/vk_pipeline.cpp
        rhi/vulkan/vk_image.cpp
        rhi/vulkan/vk_errors.cpp
        rhi/vulkan/vk_enum_conversions.cpp
        rhi/vulkan/vk_descriptor.cpp
        rhi/vulkan/vk_query.cpp
        rhi/vulkan/vk_render_pass.cpp

        rhi/window_systems/glfw_window.cpp

        rhi/cuda/cu_errors.cpp
        rhi/cuda/cu_device.cpp
        rhi/cuda/cu_enum_conversions.cpp

        text/detail/imgui_glfw_keymap.cpp
        text/imgui_backend.cpp
        text/imgui_shader.cpp
        text/text_renderer.cpp
        text/monospace_shader.cpp
        text/monospace_text_renderer.cpp

        backends/backend_tokens.cpp
        backends/format_conversions.cpp
        backends/rhi_geometry_layer.cpp
        backends/rhi_geometry_shader.cpp
        backends/rhi_image_layer.cpp
        backends/rhi_image_shader.cpp
        backends/rhi_lut_layer.cpp
        backends/rhi_lut_shader.cpp
        backends/rhi_render_backend.cpp
        backends/rhi_text_layer.cpp
        backends/wsi_present_backend.cpp
    )

# Direct-display ownership classification + acquisition is the one genuinely
# platform-specific corner of the RHI: WinRT (Windows.Devices.Display.Core +
# VK_NV_acquire_winrt_display) on Windows, libdrm (VK_EXT_acquire_drm_display)
# on Linux. See src/rhi/display_systems/display_ownership.hpp.
if(WIN32)
    target_sources(holoviz PRIVATE rhi/display_systems/display_ownership_win32.cpp)
    # C++/WinRT projection over the OS WinRT APIs (headers ship with the Windows SDK).
    target_link_libraries(holoviz PRIVATE windowsapp.lib)
elseif(UNIX)
    target_sources(holoviz PRIVATE rhi/display_systems/display_ownership_drm.cpp)
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
        target_sources(holoviz PRIVATE rhi/display_systems/display_ownership_x11.cpp)
        target_link_libraries(holoviz PRIVATE PkgConfig::X11Xrandr)
        target_compile_definitions(holoviz PRIVATE
            VK_USE_PLATFORM_XLIB_XRANDR_EXT
            HOLOVIZ_HAS_XLIB_XRANDR
        )
    endif()
endif()

# Dear ImGui is compiled directly into holoviz instead of a static archive: object files are not
# subject to --exclude-libs, so with IMGUI_API pinned to default visibility the ImGui entry
# points stay exported for the debug UI-callback mode (<holoviz/text/imgui_ui.hpp>) while every
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
elseif(MSVC AND HOLOVIZ_BUILD_SHARED)
    target_compile_definitions(holoviz PRIVATE "IMGUI_API=__declspec(dllexport)")
endif()
unset(_holoviz_imgui_source_dir)

target_compile_definitions(holoviz
    PRIVATE
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        GLFW_INCLUDE_NONE
    )

# Internal headers (rhi/detail/*_impl.hpp) are included as "rhi/detail/..." from
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
        slang::slang
        glfw
        # Pinned Khronos headers only — no loader link; volk dlopens libvulkan at runtime.
        Vulkan::Headers
        Vulkan::UtilityHeaders
        volk::volk_headers
        GPUOpen::VulkanMemoryAllocator
    )

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
    EXPORT_NAME holoviz
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    VISIBILITY_INLINES_HIDDEN YES
)

# Static dependencies (Slang, GLFW, volk, VMA) must not re-export their symbols through this
# DSO; without this the linker would surface every archive symbol it pulled in.
if(HOLOVIZ_BUILD_SHARED AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(holoviz PRIVATE
        LINKER:--exclude-libs,ALL
    )
endif()

# HoloViz is an implementation detail of the visualization module. A shared build installs its
# runtime library for the module, but neither build type exports a downstream CMake package or
# development archive.
if(HOLOVIZ_BUILD_SHARED)
    install(TARGETS holoviz
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    )
endif()
