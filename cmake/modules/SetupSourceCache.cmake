# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# This runs after project() loads the toolchain and before rapids_cpm_init().
# CPM otherwise caches OFF before the SDK supplies its default. Keep compiler
# launcher setup separate.
option(HOLOSCAN_USE_CACHE_DIR "Use CPM and compiler cache directory (set OFF for e.g. OpenEmbedded)" ON)
set(HOLOSCAN_CACHE_DIR ".cache" CACHE PATH
    "Directory for CPM and compiler caches; relative paths use the Holoscan source directory")
mark_as_advanced(HOLOSCAN_CACHE_DIR)
get_filename_component(_holoscan_cache_source_dir "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

function(check_cache_dir cache_dir_name)
    list(APPEND CMAKE_MESSAGE_CONTEXT "check_cache_dir")

    if(cache_dir_name STREQUAL "")
        message(FATAL_ERROR "cache_dir_name is empty")
    endif()

    cmake_path(IS_RELATIVE ${cache_dir_name} is_relative)

    # If the cache directory is a relative path, make it absolute path. Also normalize the path.
    if(is_relative)
        cmake_path(ABSOLUTE_PATH ${cache_dir_name} BASE_DIRECTORY "${_holoscan_cache_source_dir}" NORMALIZE)
    else()
        cmake_path(NORMAL_PATH ${cache_dir_name})
    endif()

    # If the cache directory doesn't exist, create it
    if(NOT EXISTS "${${cache_dir_name}}")
        message(STATUS "Creating folder '${${cache_dir_name}}'...")
        file(MAKE_DIRECTORY "${${cache_dir_name}}")
    endif()

    # Resolve symlinks to get the real path
    get_filename_component(RESOLVED_CACHE_DIR "${${cache_dir_name}}" REALPATH)

    # Update the cache directory with the resolved path
    set(${cache_dir_name} "${RESOLVED_CACHE_DIR}" PARENT_SCOPE)

    message(STATUS "Cache directory: ${RESOLVED_CACHE_DIR}")

endfunction()

if(HOLOSCAN_USE_CACHE_DIR)
    check_cache_dir(HOLOSCAN_CACHE_DIR)
    # A caller's normal/cache variable or environment (including OFF) takes precedence.
    # Do not overwrite an existing build tree's cached OFF during reconfiguration.
    if(NOT DEFINED CPM_SOURCE_CACHE AND NOT DEFINED ENV{CPM_SOURCE_CACHE})
        set(CPM_SOURCE_CACHE "${HOLOSCAN_CACHE_DIR}/cpm" CACHE PATH "CPM dependency source cache")
    endif()
endif()
