# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(PROJECT_NAME holoviz_unittests)

add_executable(${PROJECT_NAME})
add_executable(holoscan::viz::unittests ALIAS ${PROJECT_NAME})

# Disable clang-tidy for test files
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_CLANG_TIDY "")

find_package(CUDAToolkit REQUIRED)
find_package(Vulkan REQUIRED)

target_sources(${PROJECT_NAME}
    PRIVATE
        util/unique_value_test.cpp
        util/format_bit_depth_test.cpp
        ../../src/vulkan/format_util.cpp
    )

target_compile_definitions(${PROJECT_NAME}
    PRIVATE
        GTEST_HAS_RTTI=1
        GTEST_LANG_CXX11=1
        GTEST_HAS_EXCEPTIONS=1
    )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        GTest::gtest_main
        CUDA::cuda_driver
        Vulkan::Vulkan
        magic_enum::magic_enum
        fmt
    )

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    )

add_test(NAME unit COMMAND ${PROJECT_NAME})
