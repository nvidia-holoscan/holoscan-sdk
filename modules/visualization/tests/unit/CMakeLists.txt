# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# In the SDK-integrated build, googletest is already populated (as GTest::gtest_main) by the
# parent build's dependency setup. A top-level module build resolves an installed GTest or uses
# the shared SDK dependency recipe.
if(NOT TARGET GTest::gtest_main)
    holoscan_add_dependency(gtest_rapids COMPONENT holoscan-dependencies)
endif()

set(_holoviz_unittest_targets)

function(holoviz_add_unittest_target target_name)
    # holoviz_test_main.cpp supplies main() (so link GTest::gtest, not gtest_main): it maps
    # gtest results onto the SKIP_RETURN_CODE protocol. A skip-output regex would classify
    # the whole binary as Skipped even when another test in it failed; the exit code cannot.
    add_executable(${target_name} ${ARGN} holoviz_test_main.cpp)
    add_executable(holoviz::${target_name} ALIAS ${target_name})
    target_link_libraries(${target_name}
        PRIVATE
            holoscan::visualization::holoviz
            GTest::gtest
    )

    holoviz_copy_runtime_dlls(${target_name})
    add_test(NAME ${target_name} COMMAND ${target_name})
    # HOLOVIZ_VALIDATION=1: the whole suite runs with the Khronos validation layer and
    # the module's extra debug checks by default, so validation coverage is never lost.
    # SKIP_RETURN_CODE 77: holoviz_test_main.cpp exits 77 only when every selected test
    # skipped (a GPU-less machine reports Skipped, not a silent green); any failure exits 1
    # and a partially-skipped binary exits 0 (Passed).
    set_tests_properties(${target_name} PROPERTIES
        ENVIRONMENT "HOLOVIZ_VALIDATION=1"
        LABELS "holoviz;visualization"
        SKIP_RETURN_CODE 77
    )
    set(_holoviz_unittest_targets
        "${_holoviz_unittest_targets};${target_name}" PARENT_SCOPE)
endfunction()

holoviz_add_unittest_target(holoviz_test_logger test_logger.cpp)
holoviz_add_unittest_target(holoviz_test_device_ownership test_device_ownership.cpp)
holoviz_add_unittest_target(holoviz_test_extension_selection test_extension_selection.cpp)
# Exercises extension dependency selection without a Vulkan instance or GPU.
target_include_directories(holoviz_test_extension_selection
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
holoviz_add_unittest_target(holoviz_test_descriptor_validation test_descriptor_validation.cpp)
# The test's pipeline comes from a build-time-compiled shader, exercising the same
# generated-header path the library's built-in shaders use.
include(HolovizShaderHeader)
holoviz_generate_shader_header(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/shaders/descriptor_validation.slang.hpp"
    SLANG shaders/descriptor_validation.slang
    NAMESPACE holoviz::test::descriptor_validation
    PAIRS main=vertex_main:fragment_main
)
target_sources(holoviz_test_descriptor_validation PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/shaders/descriptor_validation.slang.hpp")
target_include_directories(holoviz_test_descriptor_validation PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated")
holoviz_add_unittest_target(holoviz_test_window_util test_window_util.cpp)
# Pure policy helpers live in the module's private src tree.
target_include_directories(holoviz_test_window_util PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

holoviz_add_unittest_target(holoviz_test_monospace_text test_monospace_text.cpp)

holoviz_add_unittest_target(holoviz_test_text_renderer_compare test_text_renderer_compare.cpp)
# The differential tests bake both renderers from the same genuinely monospace font; imgui's
# bundled Cousine is the only TTF a build is guaranteed to have fetched. Tests skip when the
# file is absent (SDK-integrated builds may resolve imgui without misc/fonts).
get_property(_holoviz_imgui_source_dir GLOBAL PROPERTY HOLOVIZ_IMGUI_SOURCE_DIR)
file(TO_CMAKE_PATH "${_holoviz_imgui_source_dir}/misc/fonts/Cousine-Regular.ttf" _holoviz_test_mono_ttf)
target_compile_definitions(holoviz_test_text_renderer_compare
    PRIVATE HOLOVIZ_TEST_MONO_TTF="${_holoviz_test_mono_ttf}")
unset(_holoviz_imgui_source_dir)
unset(_holoviz_test_mono_ttf)

holoviz_add_unittest_target(holoviz_test_imgui_renderer test_imgui_renderer.cpp)

holoviz_add_unittest_target(holoviz_test_imgui_renderer_fault_injection
    test_imgui_renderer_fault_injection.cpp)
# Reaches through the module-private impl seam (imgui/renderer_impl.hpp) to inject
# pinned-staging allocation/free failures; the main imgui renderer test file stays detail-free.
target_include_directories(holoviz_test_imgui_renderer_fault_injection
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

holoviz_add_unittest_target(holoviz_test_imgui_input test_imgui_input.cpp)
# Captures the GLFW key trampoline libholoviz registers (glfwSetKeyCallback returns the
# previous callback) to drive the backend's key handler with unmapped codes; needs the GLFW
# headers and, in static builds, the same GLFW objects libholoviz absorbed. GLFW_INCLUDE_NONE
# (as libholoviz itself compiles) keeps glfw3.h from pulling in <GL/gl.h>, which build
# environments without OpenGL headers do not have.
target_link_libraries(holoviz_test_imgui_input PRIVATE glfw)
target_compile_definitions(holoviz_test_imgui_input PRIVATE GLFW_INCLUDE_NONE)

# Regression for the sub_frame example's band sampling (examples/core/sub_frame): the
# example's shader is compiled into the test (same generated-header path the example uses, so
# the shared PushConstants layout is asserted here too) and its band pipeline is reproduced
# offscreen with the next band's CUDA upload parked. Needs a GPU with CUDA; skips otherwise.
include(HolovizShaderHeader)
holoviz_generate_shader_header(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/sub_frame/shader.slang.hpp"
    SLANG ../../examples/core/sub_frame/shader.slang
    NAMESPACE holoviz::examples::sub_frame
    PAIRS main=vertex_main:fragment_main
    PUSH_CONSTANTS_TYPE holoviz::examples::sub_frame::PushConstants
    PUSH_CONSTANTS_INCLUDE shader_interface.hpp
)
holoviz_add_unittest_target(holoviz_test_sub_frame_band_sampling
    test_sub_frame_band_sampling.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/sub_frame/shader.slang.hpp")
target_include_directories(holoviz_test_sub_frame_band_sampling
    PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/sub_frame"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/core/sub_frame")

holoviz_add_unittest_target(holoviz_test_cuda_event_ring test_cuda_event_ring.cpp)
# Exercises the examples' shared CUDA event-ring helper (a header-only example utility).
target_include_directories(holoviz_test_cuda_event_ring
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/common)

holoviz_add_unittest_target(holoviz_test_measured_query_tracker test_measured_query_tracker.cpp)
# Exercises the examples' shared GPU-query ownership helper (header-only, GPU-free).
target_include_directories(holoviz_test_measured_query_tracker
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/common)

holoviz_add_unittest_target(holoviz_test_swapchain_resize test_swapchain_resize.cpp)
holoviz_add_unittest_target(holoviz_test_gate_staleness test_gate_staleness.cpp)
holoviz_add_unittest_target(holoviz_test_destroy test_destroy.cpp)
holoviz_add_unittest_target(holoviz_test_present_timing test_present_timing.cpp)
# Reaches the swapchain fault-injection seam (vulkan/vk_swapchain_fault_injection.hpp) to
# model the presentation engine's own timing-queue-full rejection, which no driver produces on cue.
target_include_directories(holoviz_test_present_timing
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
holoviz_add_unittest_target(holoviz_test_render_pass_load test_render_pass_load.cpp)

# Pins the VK_EXT_present_timing result decoder (vulkan/vk_present_timing.hpp) with no GPU:
# the module-private decoder and enum-conversion TUs are compiled into the test (the DSO keeps
# them hidden). Header-only Vulkan/volk/VMA targets are needed because the TUs include the
# module's Vulkan headers; nothing from them is linked.
holoviz_add_unittest_target(holoviz_test_present_timing_decode
    test_present_timing_decode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/vulkan/vk_present_timing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/vulkan/vk_enum_conversions.cpp)
target_include_directories(holoviz_test_present_timing_decode
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_link_libraries(holoviz_test_present_timing_decode
    PRIVATE
        Vulkan::Headers
        volk::volk_headers
        GPUOpen::VulkanMemoryAllocator
)

holoviz_add_unittest_target(holoviz_test_display test_display.cpp)
# Exercises the module-private pure selection helpers (vk_display_util.hpp) directly.
target_include_directories(holoviz_test_display PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Drives the module-private kernel DRM ioctl client (display_systems/drm_ioctl.hpp)
    # through its ioctl seam with a fake device model: no /dev/dri node, no GPU. The
    # client TU is compiled into the test (the DSO keeps it hidden; it depends only on
    # holoviz/expected.hpp).
    holoviz_add_unittest_target(holoviz_test_drm_ioctl
        test_drm_ioctl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/display_systems/drm_ioctl.cpp)
    target_include_directories(holoviz_test_drm_ioctl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

    # Pins the direct-display surface's teardown policy through the public Surface wrapper with
    # no GPU: the module-private surface TU (vulkan/vk_display_surface.cpp) is compiled into the
    # test, which supplies fake volk entry points, a fake close_acquisition_resources() and a
    # pipe fd in place of the DRM fd. Header-only Vulkan/volk/VMA targets are needed because the
    # TU includes the module's Vulkan headers; nothing from them is linked.
    holoviz_add_unittest_target(holoviz_test_display_surface_teardown
        test_display_surface_teardown.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/vulkan/vk_display_surface.cpp)
    target_include_directories(holoviz_test_display_surface_teardown
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
    target_link_libraries(holoviz_test_display_surface_teardown
        PRIVATE
            Vulkan::Headers
            volk::volk_headers
            GPUOpen::VulkanMemoryAllocator
    )
endif()

add_custom_target(holoscan_visualization_test DEPENDS ${_holoviz_unittest_targets})
unset(_holoviz_unittest_targets)
