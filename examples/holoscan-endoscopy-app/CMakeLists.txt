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

# CUDA_STANDARD 17 is supported from CMAKE 3.18
# : https://cmake.org/cmake/help/v3.18/prop_tgt/CUDA_STANDARD.html
cmake_minimum_required(VERSION 3.18)

project(holoscan-endoscopy-app VERSION 0.0.1 DESCRIPTION "Holoscan endoscopy app" LANGUAGES CXX)

################################################################################
# Find holoscan package
################################################################################

if(NOT HOLOSCAN_SDK_PATH)
    message(FATAL_ERROR "Please set 'HOLOSCAN_SDK_PATH' variable in CMake")
else()
    list(APPEND CMAKE_MODULE_PATH "${HOLOSCAN_SDK_PATH}/cmake")
    list(APPEND CMAKE_PREFIX_PATH "${HOLOSCAN_SDK_PATH}")
    message("HOLOSCAN_SDK_PATH is set to ${HOLOSCAN_SDK_PATH}")
endif()

### FIND holoscan-embedded
find_package(holoscan-embedded CONFIG REQUIRED)

################################################################################
# Add executable: holoscan-endoscopy-app
################################################################################

add_executable(holoscan-endoscopy-app main.cpp)

set_target_properties(holoscan-endoscopy-app
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(holoscan-endoscopy-app PRIVATE holoscan-embedded::holoscan-embedded)
