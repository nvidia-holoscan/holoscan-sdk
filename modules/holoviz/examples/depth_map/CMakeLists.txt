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

set(PROJECT_NAME holoviz_depth_map)

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

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
    PRIVATE
        Main.cpp
    )

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${stb_SOURCE_DIR}
    )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        holoscan::viz
        holoscan::viz::imgui
    )
