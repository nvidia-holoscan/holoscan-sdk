# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(PROJECT_NAME holoviz_functionaltests)

# fetch the dependencies
include(FetchContent)

# stb
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG af1a5bc352164740c1cc1354942b1c6b72eacb8a
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(stb)

find_package(CUDAToolkit REQUIRED)
find_package(X11 REQUIRED)
find_package(Vulkan REQUIRED)

add_executable(${PROJECT_NAME})
add_executable(holoscan::viz::functionaltests ALIAS ${PROJECT_NAME})

# Copy the external font fixture into the build tree so tests can resolve it
# relative to CMAKE_BINARY_DIR regardless of the source/build directory layout.
file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/fonts/Roboto-Bold.ttf
    DESTINATION ${CMAKE_BINARY_DIR}/tests/holoviz/fonts
)

# Disable clang-tidy for test files
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "")

target_sources(${PROJECT_NAME}
    PRIVATE
        bit_depth_quantization_warning_test.cpp
        callback_test.cpp
        camera_pose_test.cpp
        display_event_fallback_test.cpp
        geometry_layer_test.cpp
        test_fixture.cpp
        im_gui_layer_test.cpp
        image_layer_test.cpp
        init_test.cpp
        instance_test.cpp
        layer_test.cpp
        present_mode_test.cpp
        present_wait_fallback_test.cpp
        read_framebuffer_test.cpp
        render_mode_test.cpp
        surface_format_test.cpp
        vsync_test.cpp

        ../../src/cuda/cuda_service.cpp
    )

target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        GTEST_HAS_RTTI=1
        GTEST_LANG_CXX11=1
        GTEST_HAS_EXCEPTIONS=1
    )

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${stb_SOURCE_DIR}
    )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        glfw
        X11::X11
        Vulkan::Vulkan
        holoscan::viz
        holoscan::viz::imgui
        GTest::gtest_main
        CUDA::cuda_driver
        holoscan::logger
    )

add_test(NAME functional COMMAND ${PROJECT_NAME})
set_tests_properties(functional PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
