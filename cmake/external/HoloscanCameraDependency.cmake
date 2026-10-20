# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

get_property(_cmake_role GLOBAL PROPERTY CMAKE_ROLE)

# Build the in-development Holoscan Camera module as an optional SDK
# subproject. Camera owns its install/export surface; this integration provides
# source-tree build targets only. When executed directly in CMake script mode,
# fetch sources into the working directory without configuring Camera or
# loading SDK build tools.

# Authentication is supplied by Git's environment or credential helper.
set(HOLOSCAN_CAMERA_GIT_REPOSITORY
    "https://github.com/nvidia-holoscan/holoscan-camera.git"
    CACHE STRING "Git repository used to fetch the Holoscan Camera module")
set(HOLOSCAN_CAMERA_GIT_TAG
    "97bd727032c861af02aa184ea3f8209beac3a29c"
    CACHE STRING "Git commit, tag, or branch used to fetch the Holoscan Camera module")

if(NOT HOLOSCAN_CAMERA_GIT_REPOSITORY OR NOT HOLOSCAN_CAMERA_GIT_TAG)
    message(FATAL_ERROR "Holoscan Camera requires a nonempty Git repository and revision")
endif()

if(_cmake_role STREQUAL "SCRIPT")
    include(FetchContent)

    FetchContent_Populate(holoscan_camera
        GIT_REPOSITORY "${HOLOSCAN_CAMERA_GIT_REPOSITORY}"
        GIT_TAG "${HOLOSCAN_CAMERA_GIT_TAG}"
        GIT_SHALLOW FALSE
        SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/holoscan-camera-src"
    )

    message(STATUS "Holoscan Camera source: ${holoscan_camera_SOURCE_DIR}")
    unset(_cmake_role)
    return()
endif()

include(${rapids-cmake-dir}/cpm/find.cmake)

if(holoscan_camera_FOUND OR TARGET holoscan::v4l2_capture_op)
    message(STATUS "Found Holoscan Camera, skipping source build")
    unset(_cmake_role)
    return()
endif()

set(HOLOSCAN_CAMERA_VERSION 0.1.0)
set(HOLOSCAN_CAMERA_BUILD_TESTING ${HOLOSCAN_BUILD_TESTS} CACHE BOOL
    "Build the Holoscan Camera module's CTest suites" FORCE)

rapids_cpm_find(holoscan_camera ${HOLOSCAN_CAMERA_VERSION}
    GLOBAL_TARGETS
        holoscan::v4l2_capture_op

    CPM_ARGS
        GIT_REPOSITORY "${HOLOSCAN_CAMERA_GIT_REPOSITORY}"
        GIT_TAG "${HOLOSCAN_CAMERA_GIT_TAG}"
        GIT_SHALLOW FALSE
        GIT_PROGRESS TRUE

    OPTIONS
        "HOLOSCAN_CAMERA_BUILD_ALL ON"
)

unset(_cmake_role)
