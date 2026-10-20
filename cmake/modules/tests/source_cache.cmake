# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

if(NOT DEFINED HOLOSCAN_CACHE_CASE)
    foreach(cache_case default relative cache-path cache-off normal-path normal-off
                       environment-path environment-off disabled precedence)
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E env --unset=CPM_SOURCE_CACHE
                "${CMAKE_COMMAND}"
                "-DHOLOSCAN_MODULE_DIR=${HOLOSCAN_MODULE_DIR}"
                "-DHOLOSCAN_CACHE_TEST_ROOT=${HOLOSCAN_CACHE_TEST_ROOT}"
                "-DHOLOSCAN_CACHE_CASE=${cache_case}"
                -P "${CMAKE_CURRENT_LIST_FILE}"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE output ERROR_VARIABLE error)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "Source-cache case ${cache_case} failed:\n${output}\n${error}")
        endif()
    endforeach()
    return()
endif()

set(HOLOSCAN_CACHE_DIR "${HOLOSCAN_CACHE_TEST_ROOT}/${HOLOSCAN_CACHE_CASE}")
set(expected_cache "${HOLOSCAN_CACHE_DIR}/cpm")
if(HOLOSCAN_CACHE_CASE STREQUAL "relative")
    get_filename_component(sdk_source "${HOLOSCAN_MODULE_DIR}/../.." ABSOLUTE)
    file(RELATIVE_PATH HOLOSCAN_CACHE_DIR "${sdk_source}" "${HOLOSCAN_CACHE_DIR}")
elseif(HOLOSCAN_CACHE_CASE MATCHES "^(cache|normal)-(path|off)$")
    if(CMAKE_MATCH_2 STREQUAL "off")
        set(expected_cache OFF)
    else()
        set(expected_cache "${HOLOSCAN_CACHE_TEST_ROOT}/caller-cache")
    endif()
    if(CMAKE_MATCH_1 STREQUAL "cache")
        set(CPM_SOURCE_CACHE "${expected_cache}" CACHE PATH "Explicit caller override")
    else()
        set(CPM_SOURCE_CACHE "${expected_cache}")
    endif()
elseif(HOLOSCAN_CACHE_CASE MATCHES "^environment-")
    if(HOLOSCAN_CACHE_CASE STREQUAL "environment-off")
        set(environment_cache OFF)
    else()
        set(environment_cache "${HOLOSCAN_CACHE_TEST_ROOT}/environment-cache")
    endif()
    set(ENV{CPM_SOURCE_CACHE} "${environment_cache}")
elseif(HOLOSCAN_CACHE_CASE STREQUAL "disabled")
    set(HOLOSCAN_USE_CACHE_DIR OFF)
elseif(HOLOSCAN_CACHE_CASE STREQUAL "precedence")
    set(CPM_SOURCE_CACHE "${expected_cache}" CACHE PATH "Explicit caller override")
    set(environment_cache OFF)
    set(ENV{CPM_SOURCE_CACHE} "${environment_cache}")
endif()

# Resolve paths from the module location, independently of the caller's project.
include("${HOLOSCAN_MODULE_DIR}/SetupSourceCache.cmake")
if(HOLOSCAN_CACHE_CASE MATCHES "^environment-" OR HOLOSCAN_CACHE_CASE STREQUAL "disabled")
    if(DEFINED CPM_SOURCE_CACHE)
        message(FATAL_ERROR "SDK supplied a CPM default despite a caller override")
    endif()
elseif(NOT CPM_SOURCE_CACHE STREQUAL expected_cache)
    message(FATAL_ERROR "Expected '${expected_cache}', got '${CPM_SOURCE_CACHE}'")
endif()
if(DEFINED environment_cache)
    if(NOT "$ENV{CPM_SOURCE_CACHE}" STREQUAL environment_cache)
        message(FATAL_ERROR "SDK changed the caller's CPM environment")
    endif()
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
    message(FATAL_ERROR "SDK unexpectedly changed the CPM environment")
endif()
