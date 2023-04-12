# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
find_package(Vulkan REQUIRED)

add_executable(${PROJECT_NAME})
add_executable(holoscan::viz::functionaltests ALIAS ${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE
        headless_fixture.cpp
        geometry_layer_test.cpp
        im_gui_layer_test.cpp
        image_layer_test.cpp
        init_test.cpp
        layer_test.cpp

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
        Vulkan::Vulkan
        X11::X11
        holoscan::viz
        holoscan::viz::imgui
        GTest::gtest_main
        CUDA::cuda_driver
        holoscan::logger
    )

add_test(NAME functional COMMAND ${PROJECT_NAME})
