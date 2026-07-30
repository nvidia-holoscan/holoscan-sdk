# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CCCL is a dependency of MatX and RMM.
set(CCCL_REQUESTED_VERSION 3.2.0)

# Try to find system CCCL in the build container first.
# Subsequent RAPIDS CPM components (MatX) can get confused
# if both a compatible system and cache CCCL are present.
find_package(CCCL ${CCCL_REQUESTED_VERSION} PATHS /usr/local/cuda/lib64/cmake QUIET)

if(NOT CCCL_FOUND)
    # Install CCCL headers under include/cccl/ to match the expected layout from cccl-config.cmake.
    # Required in order for CCCL CMake import strategy to parse includes and set user "-I" priority
    # such that custom CCCL headers are included before older CUDA Toolkit CCCL packages at compile time.
    set(_holoscan_cccl_saved_includedir "${CMAKE_INSTALL_INCLUDEDIR}")
    set(CMAKE_INSTALL_INCLUDEDIR "include/cccl" CACHE PATH "Installation directory for header files" FORCE)

    rapids_cpm_find(CCCL ${CCCL_REQUESTED_VERSION}
        GLOBAL_TARGETS CCCL CCCL::CCCL CCCL::CUB CCCL::libcudacxx
        BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports

        CPM_ARGS
            GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
            GIT_TAG v${CCCL_REQUESTED_VERSION}
            GIT_SHALLOW TRUE
            EXCLUDE_FROM_ALL
            SOURCE_DIR _deps/cccl-src
            UPDATE_DISCONNECTED TRUE

        OPTIONS "CCCL_TOPLEVEL_PROJECT OFF"
                "CCCL_ENABLE_INSTALL_RULES ON"
                "CUB_ENABLE_INSTALL_RULES ON"
                "Thrust_ENABLE_INSTALL_RULES ON"
                "cudax_ENABLE_INSTALL_RULES ON"
                "libcudacxx_ENABLE_INSTALL_RULES ON"
    )

    set(CMAKE_INSTALL_INCLUDEDIR "${_holoscan_cccl_saved_includedir}" CACHE PATH "Installation directory for header files" FORCE)
endif()
