# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_BUILD_TREE_FIXTURE_SOURCE_DIR
        HOLOSCAN_BUILD_TREE_FIXTURE_WORK_DIR
        HOLOSCAN_BUILD_TREE_FIND_MODE
        HOLOSCAN_GENERATOR)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

set(rejection_marker "HOLOSCAN_RAW_BUILD_TREE_PACKAGE_UNSUPPORTED")
set(build_tree_config "${HOLOSCAN_BUILD_DIR}/holoscan-config.cmake")
set(build_tree_version "${HOLOSCAN_BUILD_DIR}/holoscan-config-version.cmake")
set(required_guidance
    "${rejection_marker}"
    "raw build tree is not a supported CMake package"
    "Install Holoscan into a prefix"
    "CMAKE_PREFIX_PATH"
    "holoscan_DIR"
)
foreach(build_tree_entry IN ITEMS "${build_tree_config}" "${build_tree_version}")
    if(NOT EXISTS "${build_tree_entry}")
        message(FATAL_ERROR "Raw build-tree package entry is missing: ${build_tree_entry}")
    endif()
endforeach()
file(READ "${build_tree_config}" build_tree_config_text)
file(READ "${build_tree_version}" build_tree_version_text)
if(NOT build_tree_config_text STREQUAL build_tree_version_text)
    message(FATAL_ERROR
        "Raw build-tree config and version entries must carry identical fail-fast guidance")
endif()
foreach(build_tree_entry IN ITEMS "${build_tree_config}" "${build_tree_version}")
    file(READ "${build_tree_entry}" build_tree_entry_text)
    foreach(guidance_fragment IN LISTS required_guidance)
        string(FIND "${build_tree_entry_text}" "${guidance_fragment}" guidance_position)
        if(guidance_position EQUAL -1)
            message(FATAL_ERROR
                "Raw build-tree package entry lacks required guidance '${guidance_fragment}': "
                "${build_tree_entry}")
        endif()
    endforeach()
endforeach()

set(consumer_build_dir
    "${HOLOSCAN_BUILD_TREE_FIXTURE_WORK_DIR}/${HOLOSCAN_BUILD_TREE_FIND_MODE}")
file(REMOVE_RECURSE "${consumer_build_dir}")

set(configure_command
    "${CMAKE_COMMAND}"
    -E env
    "HOLOSCAN_BUILD_TREE_ROOT=${HOLOSCAN_BUILD_DIR}"
    "HOLOSCAN_FIND_MODE=${HOLOSCAN_BUILD_TREE_FIND_MODE}"
    "${CMAKE_COMMAND}"
    -S "${HOLOSCAN_BUILD_TREE_FIXTURE_SOURCE_DIR}"
    -B "${consumer_build_dir}"
    -G "${HOLOSCAN_GENERATOR}"
)
if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
endif()
if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
endif()
if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
    list(APPEND configure_command "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
endif()
if(DEFINED HOLOSCAN_CXX_COMPILER AND NOT HOLOSCAN_CXX_COMPILER STREQUAL "")
    list(APPEND configure_command "-DCMAKE_CXX_COMPILER:FILEPATH=${HOLOSCAN_CXX_COMPILER}")
endif()
if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    list(APPEND configure_command "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${HOLOSCAN_TOOLCHAIN_FILE}")
endif()
if(DEFINED HOLOSCAN_MSVC_RUNTIME_LIBRARY AND NOT HOLOSCAN_MSVC_RUNTIME_LIBRARY STREQUAL "")
    list(APPEND configure_command
        "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${HOLOSCAN_MSVC_RUNTIME_LIBRARY}")
endif()
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND configure_command "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
endif()

execute_process(
    COMMAND ${configure_command}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_stdout
    ERROR_VARIABLE configure_stderr
)
set(configure_output "${configure_stdout}\n${configure_stderr}")
if(configure_result EQUAL 0)
    message(FATAL_ERROR
        "Raw build-tree package lookup unexpectedly succeeded for "
        "${HOLOSCAN_BUILD_TREE_FIND_MODE}")
endif()
foreach(guidance_fragment IN LISTS required_guidance)
    string(FIND "${configure_output}" "${guidance_fragment}" guidance_position)
    if(guidance_position EQUAL -1)
        message(FATAL_ERROR
            "Raw build-tree package lookup omitted required guidance '${guidance_fragment}' for "
            "${HOLOSCAN_BUILD_TREE_FIND_MODE}:\n${configure_output}")
    endif()
endforeach()
if(EXISTS "${consumer_build_dir}/holoscan-package-accepted")
    message(FATAL_ERROR
        "Raw build-tree package lookup partially accepted ${HOLOSCAN_BUILD_TREE_FIND_MODE}")
endif()
if(EXISTS "${consumer_build_dir}/_deps")
    message(FATAL_ERROR
        "Raw build-tree package lookup configured or fetched a dependency subtree")
endif()

# CMake creates its ordinary compiler/configure cache before evaluating find_package(). The
# unsupported package must not add any Holoscan acceptance or component state to that cache.
set(consumer_cache "${consumer_build_dir}/CMakeCache.txt")
if(EXISTS "${consumer_cache}")
    file(STRINGS "${consumer_cache}" holoscan_cache_entries
        REGEX "^(holoscan|HOLOSCAN|_holoscan)[^:]*:")
    foreach(holoscan_cache_entry IN LISTS holoscan_cache_entries)
        if(holoscan_cache_entry MATCHES
            "^holoscan_build_tree_package_consumer_(BINARY_DIR|IS_TOP_LEVEL|SOURCE_DIR):STATIC=")
            continue()
        endif()
        if(NOT holoscan_cache_entry STREQUAL "holoscan_DIR:PATH=holoscan_DIR-NOTFOUND")
            message(FATAL_ERROR
                "Raw build-tree package lookup mutated Holoscan package state in the consumer "
                "cache: ${holoscan_cache_entry}")
        endif()
    endforeach()
endif()
