# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The Holoscan Sensor Bridge project provides low-level Hololink sensor APIs.
# See: https://www.nvidia.com/en-us/technologies/holoscan-sensor-bridge/

if(hololink_FOUND OR TARGET hololink::hololink)
    message(STATUS "Found unexpected Holoscan Sensor Bridge, skipping build")
    return()
endif()

# Holoscan Sensor Bridge 2.6.0
set(HOLOLINK_VERSION 2.6.0)

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
        PATCH_COMMAND sh -c "patch -Np1 -i ${CMAKE_CURRENT_LIST_DIR}/patches/hololink.patch" || [ $? -eq 1 ]
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE

    OPTIONS
        "HOLOLINK_BUILD_EXAMPLES ON"
        "HOLOLINK_BUILD_TESTS ON"
        EXCLUDE_FROM_ALL
)
