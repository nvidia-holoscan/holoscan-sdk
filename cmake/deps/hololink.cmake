# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# The Holoscan Sensor Bridge project provides low-level Hololink sensor APIs.
# See: https://www.nvidia.com/en-us/technologies/holoscan-sensor-bridge/

if(hololink_FOUND OR TARGET hololink::hololink)
    message(STATUS "Found unexpected Holoscan Sensor Bridge, skipping build")
    return()
endif()

# Holoscan Sensor Bridge 2.5.0 + "top of tree" continuous fixes
set(HOLOLINK_VERSION 6930609c4) # 2.5.0-PB6

rapids_cpm_find(hololink ${HOLOLINK_VERSION}
    GLOBAL_TARGETS
        hololink
        hololink::hololink
        hololink::core
        hololink::emulation
        hololink::emulation::coe
        hololink::emulation::roce
        hololink::emulation::sensors
        hololink::emulation::sensors::argus
        hololink::emulation::sensors::roce
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports

    CPM_ARGS
        GITHUB_REPOSITORY nvidia-holoscan/holoscan-sensor-bridge
        GIT_TAG ${HOLOLINK_VERSION}
        PATCH_COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/patches/hololink.patch
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE

    OPTIONS
        "HOLOLINK_BUILD_EXAMPLES ON"
        "HOLOLINK_BUILD_TESTS ON"
        EXCLUDE_FROM_ALL
)
