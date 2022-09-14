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

list(APPEND CMAKE_MESSAGE_CONTEXT "cache")

message(STATUS "Configuring Cache for CPM and CCache")

function(check_cache_dir cache_dir_name)
    list(APPEND CMAKE_MESSAGE_CONTEXT "check_cache_dir")

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

    # Update the cache directory
    set(${cache_dir_name} "${${cache_dir_name}}" PARENT_SCOPE)

    message(STATUS "Cache directory: ${${cache_dir_name}}")

    # If CMAKE_FIND_ROOT_PATH is set, add the cache directory to it
    if(NOT "${CMAKE_FIND_ROOT_PATH}" STREQUAL "")
        message(STATUS "Cache directory '${${cache_dir_name}}'(by ${cache_dir_name}) "
            " is not accessible from CMAKE_FIND_ROOT_PATH('${CMAKE_FIND_ROOT_PATH}'). "
            " Appending the cache path to CMAKE_FIND_ROOT_PATH for CPM to find the downloaded packages...")

        list(APPEND CMAKE_FIND_ROOT_PATH "${${cache_dir_name}}")
        set(CMAKE_FIND_ROOT_PATH "${CMAKE_FIND_ROOT_PATH}" PARENT_SCOPE)
    endif()
endfunction()

function(configure_cpm cache_dir_name)
    list(APPEND CMAKE_MESSAGE_CONTEXT "configure_cpm")

    # Set the CPM_SOURCE_CACHE environment
    set(ENV{CPM_SOURCE_CACHE} "${${cache_dir_name}}/cpm")

    message(STATUS "CPM source cache directory: $ENV{CPM_SOURCE_CACHE}")
endfunction()

# This function requires the following arguments:
# - LANGUAGE: The language to be configured
# - TEMP_DIR: The temporary directory to be used for the cache (default: /tmp)
function(gen_ccache_launcher)
    list(APPEND CMAKE_MESSAGE_CONTEXT "gen_ccache_launcher")
    set(options "")
    set(one_value LANGUAGE TEMP_DIR)
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

    if(NOT DEFINED GEN_BIN_TEMP_DIR)
        # TODO(gbae): Update this implementation if we support other platforms (such as Windows)
        set(GEN_BIN_TEMP_DIR "/tmp")
    endif()

    # Set GEN_BIN_LANGUAGE to the uppercase of LANGUAGE.
    string(TOUPPER ${GEN_BIN_LANGUAGE} GEN_BIN_LANGUAGE)

    # Set GEN_BIN_LANGUAGE_LOWERCASE to the lowercase of LANGUAGE.
    string(TOLOWER ${GEN_BIN_LANGUAGE} GEN_BIN_LANGUAGE_LOWERCASE)

    # Set CCACHE_COMPILERTYPE explicitly to make it work with arbitrary compiler executable names (e.g. c++ in Conda build).
    if("${CMAKE_${GEN_BIN_LANGUAGE}_COMPILER_ID}" STREQUAL "GNU")
        set(CCACHE_COMPILERTYPE "gcc")
    elseif("${CMAKE_${GEN_BIN_LANGUAGE}_COMPILER_ID}" STREQUAL "Clang")
        set(CCACHE_COMPILERTYPE "clang")
    elseif("${CMAKE_${GEN_BIN_LANGUAGE}_COMPILER_ID}" STREQUAL "NVIDIA")
        set(CCACHE_COMPILERTYPE "nvcc")
    else()
        set(CCACHE_COMPILERTYPE "auto")
    endif()

    set(GEN_BIN_EXECUTABLE_PATH "${CMAKE_CURRENT_BINARY_DIR}/launch_ccache_${GEN_BIN_LANGUAGE_LOWERCASE}")

    configure_file("${CMAKE_CURRENT_LIST_DIR}/ccache/launch_ccache.sh.in" "${GEN_BIN_TEMP_DIR}/launch_ccache_${GEN_BIN_LANGUAGE_LOWERCASE}" @ONLY)

    # Since 'file(CHMOD)' is supported since later CMake versions(>=3.19), we use 'file(COPY)' instead to add permission(+x), using the temporary file
    file(
        COPY "${GEN_BIN_TEMP_DIR}/launch_ccache_${GEN_BIN_LANGUAGE_LOWERCASE}" # ${GEN_BIN_EXECUTABLE_PATH}
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )

    # Expose the executable path to the caller
    set(GEN_BIN_EXECUTABLE_PATH "${GEN_BIN_EXECUTABLE_PATH}" PARENT_SCOPE)
endfunction()

function(configure_ccache cache_dir_name)
    list(APPEND CMAKE_MESSAGE_CONTEXT "configure_ccache")

    find_program(CCACHE_BIN_PATH ccache DOC "Path of ccache executable")

    if(NOT CCACHE_BIN_PATH)
        message(WARNING "Option 'HOLOSCAN_USE_CCACHE' is set to TRUE but cannot find ccacche executable. "
            "Please check ccache installation. Skipping using CCache")
        set(HOLOSCAN_USE_CCACHE_SKIPPED TRUE PARENT_SCOPE)
        return()
    endif()

    set(CCACHE_DIR "${${cache_dir_name}}/ccache")

    message(STATUS "CCache executable path: ${CCACHE_BIN_PATH}")
    message(STATUS "CCache data directory : ${CCACHE_DIR}")

    # Use generated CCache config file
    configure_file("${CMAKE_CURRENT_LIST_DIR}/ccache/ccache.conf.in" "${CCACHE_DIR}/ccache.conf" @ONLY)
    set(CCACHE_CONFIGPATH "${CCACHE_DIR}/ccache.conf")

    # Set CCACHE_BASEDIR to use relative paths for the different working directories
    set(CCACHE_BASEDIR "${PROJECT_SOURCE_DIR}")

    gen_ccache_launcher(LANGUAGE C)
    set(CMAKE_C_COMPILER_LAUNCHER "${GEN_BIN_EXECUTABLE_PATH}" PARENT_SCOPE)
    gen_ccache_launcher(LANGUAGE CXX)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${GEN_BIN_EXECUTABLE_PATH}" PARENT_SCOPE)
    gen_ccache_launcher(LANGUAGE CUDA)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${GEN_BIN_EXECUTABLE_PATH}" PARENT_SCOPE)

    # Expose CCACHE_DIR to the caller
    set(CCACHE_DIR "${CCACHE_DIR}" PARENT_SCOPE)
endfunction()

# ##############################################################################
if(${HOLOSCAN_CACHE_DIR} STREQUAL " ")
    message(STATUS " HOLOSCAN_CACHE_DIR is not set. Defaulting to '${CMAKE_SOURCE_DIR}/.cache' ... ")
    set(HOLOSCAN_CACHE_DIR "${CMAKE_SOURCE_DIR}/.cache" CACHE)
endif()

check_cache_dir(HOLOSCAN_CACHE_DIR)
configure_cpm(HOLOSCAN_CACHE_DIR)

if(HOLOSCAN_USE_CCACHE)
    configure_ccache(HOLOSCAN_CACHE_DIR)
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
