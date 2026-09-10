# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# In the SDK-integrated build, googletest is already populated (as GTest::gtest_main) by the
# parent build's CPM setup. A top-level module build resolves an installed GTest or fetches it.
if(NOT TARGET GTest::gtest_main)
    find_package(GTest CONFIG QUIET)
endif()
if(NOT TARGET GTest::gtest_main)
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(googletest)
endif()

function(holoviz_add_unittest_target target_name)
    # holoviz_test_main.cpp supplies main() (so link GTest::gtest, not gtest_main): it maps
    # gtest results onto the SKIP_RETURN_CODE protocol. A skip-output regex would classify
    # the whole binary as Skipped even when another test in it failed; the exit code cannot.
    add_executable(${target_name} ${ARGN} holoviz_test_main.cpp)
    add_executable(holoviz::${target_name} ALIAS ${target_name})
    target_link_libraries(${target_name}
        PRIVATE
            holoviz::holoviz
            GTest::gtest
    )

    holoviz_copy_runtime_dlls(${target_name})
    add_test(NAME ${target_name} COMMAND ${target_name})
    # HOLOVIZ_RHI_VALIDATION=1: the whole suite runs with the Khronos validation layer and
    # the module's extra debug checks by default, so validation coverage is never lost.
    # SKIP_RETURN_CODE 77: holoviz_test_main.cpp exits 77 only when every selected test
    # skipped (a GPU-less machine reports Skipped, not a silent green); any failure exits 1
    # and a partially-skipped binary exits 0 (Passed).
    set_tests_properties(${target_name} PROPERTIES
        ENVIRONMENT "HOLOVIZ_RHI_VALIDATION=1"
        LABELS "holoviz"
        SKIP_RETURN_CODE 77
    )
endfunction()

holoviz_add_unittest_target(holoviz_test_layer_desc test_layer_desc.cpp)
holoviz_add_unittest_target(holoviz_test_logger test_logger.cpp)
holoviz_add_unittest_target(holoviz_test_rhi_device_ownership test_rhi_device_ownership.cpp)
holoviz_add_unittest_target(holoviz_test_rhi_descriptor_validation test_rhi_descriptor_validation.cpp)
holoviz_add_unittest_target(holoviz_test_render_backend test_render_backend.cpp)
holoviz_add_unittest_target(holoviz_test_sink_desc test_sink_desc.cpp)
holoviz_add_unittest_target(holoviz_test_slang_compiler test_slang_compiler.cpp)

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

holoviz_add_unittest_target(holoviz_test_imgui_backend test_imgui_backend.cpp)

holoviz_add_unittest_target(holoviz_test_imgui_backend_fault_injection
    test_imgui_backend_fault_injection.cpp)
# Reaches through the module-private impl seam (text/imgui_backend_impl.hpp) to inject
# pinned-staging allocation/free failures; the main imgui backend test file stays detail-free.
target_include_directories(holoviz_test_imgui_backend_fault_injection
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

holoviz_add_unittest_target(holoviz_test_imgui_input test_imgui_input.cpp)
# Captures the GLFW key trampoline libholoviz registers (glfwSetKeyCallback returns the
# previous callback) to drive the backend's key handler with unmapped codes; needs the GLFW
# headers and, in static builds, the same GLFW objects libholoviz absorbed. GLFW_INCLUDE_NONE
# (as libholoviz itself compiles) keeps glfw3.h from pulling in <GL/gl.h>, which build
# environments without OpenGL headers do not have.
target_link_libraries(holoviz_test_imgui_input PRIVATE glfw)
target_compile_definitions(holoviz_test_imgui_input PRIVATE GLFW_INCLUDE_NONE)

holoviz_add_unittest_target(holoviz_test_cuda_event_ring test_cuda_event_ring.cpp)
# Exercises the examples' shared CUDA event-ring helper (a header-only example utility).
target_include_directories(holoviz_test_cuda_event_ring
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/common)

holoviz_add_unittest_target(holoviz_test_measured_query_tracker test_measured_query_tracker.cpp)
# Exercises the examples' shared GPU-query ownership helper (header-only, GPU-free).
target_include_directories(holoviz_test_measured_query_tracker
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/common)

holoviz_add_unittest_target(holoviz_test_rhi_swapchain_resize test_rhi_swapchain_resize.cpp)
holoviz_add_unittest_target(holoviz_test_rhi_gate_staleness test_rhi_gate_staleness.cpp)
holoviz_add_unittest_target(holoviz_test_rhi_destroy test_rhi_destroy.cpp)
holoviz_add_unittest_target(holoviz_test_enums test_enums.cpp)

holoviz_add_unittest_target(holoviz_test_rhi_display test_rhi_display.cpp)
# Exercises the module-private pure selection helpers (vk_display_util.hpp) directly.
target_include_directories(holoviz_test_rhi_display PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

holoviz_add_unittest_target(holoviz_test_wsi_present_backend test_wsi_present_backend.cpp)
# Exercises the module-private frame table / token payloads / clamp helpers directly.
target_include_directories(holoviz_test_wsi_present_backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

holoviz_add_unittest_target(holoviz_test_wsi_present_fault_injection
    test_wsi_present_fault_injection.cpp)
# Reaches through the module-private swapchain fault-injection seam
# (rhi/vulkan/vk_swapchain_fault_injection.hpp) to drive deterministic present/recreation
# failures; the main WSI test file stays injection-free.
target_include_directories(holoviz_test_wsi_present_fault_injection
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

holoviz_add_unittest_target(holoviz_test_rhi_render_backend test_rhi_render_backend.cpp)
# Uses the concrete backend's headless_target() readback hook and the token payloads.
target_include_directories(holoviz_test_rhi_render_backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

holoviz_add_unittest_target(holoviz_test_rhi_backend_text test_rhi_backend_text.cpp)
target_include_directories(holoviz_test_rhi_backend_text PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

holoviz_add_unittest_target(holoviz_test_rhi_backend_geometry test_rhi_backend_geometry.cpp)
target_include_directories(holoviz_test_rhi_backend_geometry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

holoviz_add_unittest_target(holoviz_test_rhi_backend_image test_rhi_backend_image.cpp)
target_include_directories(holoviz_test_rhi_backend_image PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

holoviz_add_unittest_target(holoviz_test_rhi_backend_lut test_rhi_backend_lut.cpp)
target_include_directories(holoviz_test_rhi_backend_lut PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
