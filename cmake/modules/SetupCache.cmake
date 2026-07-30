# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MESSAGE_CONTEXT "cache")

message(STATUS "Configuring Cache for CPM and Compiler Cache (SCCache)")

function(check_cache_dir cache_dir_name)
    list(APPEND CMAKE_MESSAGE_CONTEXT "check_cache_dir")

    if(cache_dir_name STREQUAL "")
        message(FATAL_ERROR "cache_dir_name is empty")
    endif()

    cmake_path(IS_RELATIVE ${cache_dir_name} is_relative)

    # If the cache directory is a relative path, make it absolute path. Also normalize the path.
    if(is_relative)
        cmake_path(ABSOLUTE_PATH ${cache_dir_name} BASE_DIRECTORY ${CMAKE_SOURCE_DIR} NORMALIZE)
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

    # If CMAKE_FIND_ROOT_PATH is set, add the cache directory to it
    if(NOT "${CMAKE_FIND_ROOT_PATH}" STREQUAL "")
        message(STATUS "Cache directory '${RESOLVED_CACHE_DIR}'(by ${cache_dir_name}) "
            " is not accessible from CMAKE_FIND_ROOT_PATH('${CMAKE_FIND_ROOT_PATH}'). "
            " Appending the cache path to CMAKE_FIND_ROOT_PATH for CPM to find the downloaded packages...")

        list(APPEND CMAKE_FIND_ROOT_PATH "${RESOLVED_CACHE_DIR}")
        set(CMAKE_FIND_ROOT_PATH "${CMAKE_FIND_ROOT_PATH}" PARENT_SCOPE)
    endif()
endfunction()

function(configure_cpm cache_dir_name)
    list(APPEND CMAKE_MESSAGE_CONTEXT "configure_cpm")

    if(cache_dir_name STREQUAL "")
        message(FATAL_ERROR "cache_dir_name is empty")
    endif()

    # Set the CPM_SOURCE_CACHE environment
    set(ENV{CPM_SOURCE_CACHE} "${${cache_dir_name}}/cpm")

    message(STATUS "CPM source cache directory: $ENV{CPM_SOURCE_CACHE}")
endfunction()

# This function requires the following arguments:
# - LANGUAGE: The language to be configured
# - LAUNCHER_TYPE: The type of launcher (ccache or sccache)
function(gen_cache_launcher)
    list(APPEND CMAKE_MESSAGE_CONTEXT "gen_cache_launcher")
    set(options "")
    set(one_value LANGUAGE LAUNCHER_TYPE)
    set(multi_value "")
    cmake_parse_arguments(GEN_BIN "${options}" "${one_value}" "${multi_value}" ${ARGN})

    if(GEN_BIN_UNPARSED_ARGUMENTS)
        message(WARNING "There are unparsed arguments: ${GEN_BIN_UNPARSED_ARGUMENTS}")
    endif()

    if(GEN_BIN_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "There are missing values for: ${GEN_BIN_KEYWORDS_MISSING_VALUES}")
    endif()

    if(NOT DEFINED GEN_BIN_LANGUAGE)
        message(FATAL_ERROR "Language is not defined")
    endif()

    if(NOT DEFINED GEN_BIN_LAUNCHER_TYPE)
        set(GEN_BIN_LAUNCHER_TYPE "sccache")
    endif()

    # Set GEN_BIN_LANGUAGE to the uppercase of LANGUAGE.
    string(TOUPPER ${GEN_BIN_LANGUAGE} GEN_BIN_LANGUAGE)

    # Set GEN_BIN_LANGUAGE_LOWERCASE to the lowercase of LANGUAGE.
    string(TOLOWER ${GEN_BIN_LANGUAGE} GEN_BIN_LANGUAGE_LOWERCASE)

    set(GEN_BIN_EXECUTABLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/launch_${GEN_BIN_LAUNCHER_TYPE}_${GEN_BIN_LANGUAGE_LOWERCASE}")

    configure_file("${CMAKE_CURRENT_LIST_DIR}/ccache/launch_${GEN_BIN_LAUNCHER_TYPE}.sh.in" "${GEN_BIN_EXECUTABLE_PATH}" @ONLY)
    file(CHMOD "${GEN_BIN_EXECUTABLE_PATH}" PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

    # Expose the executable path to the caller
    set(GEN_BIN_EXECUTABLE_PATH "${GEN_BIN_EXECUTABLE_PATH}" PARENT_SCOPE)
endfunction()

function(configure_sccache cache_dir_name)
    list(APPEND CMAKE_MESSAGE_CONTEXT "configure_sccache")

    if(cache_dir_name STREQUAL "")
        message(FATAL_ERROR "configure_sccache: cache_dir_name is empty")
    endif()

    find_program(SCCACHE_BIN_PATH sccache DOC "Path of sccache executable")

    if(NOT SCCACHE_BIN_PATH)
        message(FATAL_ERROR "Option 'HOLOSCAN_USE_SCCACHE' is set to TRUE but cannot find sccache executable. "
            "Please install sccache.")
    endif()

    set(SCCACHE_DIR "${${cache_dir_name}}/sccache")
    set(SCCACHE_CACHE_SIZE "20G")

    # Create the sccache directory if it doesn't exist

    message(STATUS "SCCache executable path: ${SCCACHE_BIN_PATH}")
    message(STATUS "SCCache data directory : ${SCCACHE_DIR}")

    gen_cache_launcher(LANGUAGE C LAUNCHER_TYPE sccache)
    set(CMAKE_C_COMPILER_LAUNCHER "${GEN_BIN_EXECUTABLE_PATH}" PARENT_SCOPE)
    gen_cache_launcher(LANGUAGE CXX LAUNCHER_TYPE sccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${GEN_BIN_EXECUTABLE_PATH}" PARENT_SCOPE)
    gen_cache_launcher(LANGUAGE CUDA LAUNCHER_TYPE sccache)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${GEN_BIN_EXECUTABLE_PATH}" PARENT_SCOPE)

    message(STATUS "Configured sccache for C, C++, and CUDA compilation.")

    # Expose SCCACHE_DIR to the caller
    set(SCCACHE_DIR "${SCCACHE_DIR}" PARENT_SCOPE)
endfunction()

# ##############################################################################
# Default HOLOSCAN_USE_CACHE_DIR and HOLOSCAN_CACHE_DIR when this module is reused and
# variables are not set.

if(NOT DEFINED HOLOSCAN_USE_CACHE_DIR)
    set(HOLOSCAN_USE_CACHE_DIR ON)
    message(STATUS "HOLOSCAN_USE_CACHE_DIR is not set. Defaulting to '${HOLOSCAN_USE_CACHE_DIR}'")
endif()

if(NOT DEFINED HOLOSCAN_CACHE_DIR)
    set(HOLOSCAN_CACHE_DIR ".cache")
    message(STATUS "HOLOSCAN_CACHE_DIR is not set. Defaulting to '${HOLOSCAN_CACHE_DIR}'")
endif()

if(HOLOSCAN_USE_CACHE_DIR)
    check_cache_dir(HOLOSCAN_CACHE_DIR)
    configure_cpm(HOLOSCAN_CACHE_DIR)

    if(HOLOSCAN_USE_SCCACHE)
        configure_sccache(HOLOSCAN_CACHE_DIR)
    else()
        message(STATUS "SCCache is not enabled. Skipping compiler cache configuration.")
    endif()
else()
    message(STATUS "Cache disabled (HOLOSCAN_USE_CACHE_DIR=OFF). Skipping CPM and compiler cache setup.")
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
