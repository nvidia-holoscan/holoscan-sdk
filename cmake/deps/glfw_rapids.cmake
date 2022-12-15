# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_find_package.html#
include(${rapids-cmake-dir}/cpm/find.cmake)

find_package(X11)

if(NOT X11_FOUND)
    message(FATAL_ERROR "X11 not found. Please install X11 (e.g., 'sudo apt-get install libx11-dev') and try again.")
endif()

# Check for XRandR (modern resolution switching and gamma control)
if(NOT X11_Xrandr_INCLUDE_PATH)
    message(FATAL_ERROR "XRandR headers not found. Please install XRandR ('sudo apt-get install libxrandr-dev') and try again.")
endif()

# Check for Xinerama (legacy multi-monitor support)
if(NOT X11_Xrandr_INCLUDE_PATH)
    message(FATAL_ERROR "Xinerama headers not found. Please install Xinerama ('sudo apt-get install libxinerama-dev') and try again.")
endif()

# Check for Xcursor (cursor creation from RGBA images)
if(NOT X11_Xcursor_INCLUDE_PATH)
    message(FATAL_ERROR "Xcursor headers not found. Please install Xcursor ('sudo apt-get install libxcursor-dev') and try again.")
endif()

# Check for Xkb (X keyboard extension)
if(NOT X11_Xkb_INCLUDE_PATH)
    message(FATAL_ERROR "Xkb headers not found. Please install Xkb ('sudo apt-get install libxkbcommon-dev' or 'sudo apt-get install libx11-dev') and try again.")
endif()

# Check for XInput (modern HID input)
if(NOT X11_Xinput_INCLUDE_PATH)
    message(FATAL_ERROR "XInput headers not found. Please install XInput ('sudo apt-get install libxi-dev') and try again.")
endif()

rapids_cpm_find(GLFW 3.3.7
    GLOBAL_TARGETS glfw

    CPM_ARGS

    GITHUB_REPOSITORY glfw/glfw
    GIT_TAG 3.3.7
    OPTIONS
    "CXXOPTS_BUILD_EXAMPLES OFF"
    "CXXOPTS_BUILD_TESTS OFF"
    "GLFW_BUILD_TESTS OFF"
    "GLFW_BUILD_EXAMPLES OFF"
    "GLFW_BULID_DOCS OFF"
    "GLFW_INSTALL OFF"
    EXCLUDE_FROM_ALL
)

if(GLFW_ADDED)
    install(TARGETS glfw
        DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}"
        COMPONENT "holoscan-dependencies"
    )
endif()
