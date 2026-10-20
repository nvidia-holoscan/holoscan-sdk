# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

project(holoscan_runtime_installed_consumer LANGUAGES CXX)

if(NOT DEFINED HOLOSCAN_TEST_INSTALL_PREFIX OR HOLOSCAN_TEST_INSTALL_PREFIX STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_TEST_INSTALL_PREFIX must name the isolated SDK install prefix")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${HOLOSCAN_TEST_INSTALL_PREFIX}")
find_package(holoscan
    PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}"
    NO_DEFAULT_PATH
    REQUIRED
)

if(NOT TARGET holoscan::core)
    message(FATAL_ERROR "Installed package did not define holoscan::core")
endif()

# Public headers must not leak the source-private process-placement vocabulary.
file(GLOB_RECURSE HOLOSCAN_INSTALLED_PUBLIC_HEADERS
    LIST_DIRECTORIES FALSE
    "${HOLOSCAN_TEST_INSTALL_PREFIX}/include/holoscan/*.h"
    "${HOLOSCAN_TEST_INSTALL_PREFIX}/include/holoscan/*.hpp"
)
set(HOLOSCAN_PRIVATE_PROCESS_TOKENS
    GraphObjectSelector
    ProcessPlacement
    bind_process_role
    process_roles
    process_selectors
    vertex_process_placements
    launch_group
)
foreach(header IN LISTS HOLOSCAN_INSTALLED_PUBLIC_HEADERS)
    file(READ "${header}" header_content)
    foreach(token IN LISTS HOLOSCAN_PRIVATE_PROCESS_TOKENS)
        string(REGEX MATCH
            "(^|[^A-Za-z0-9_])${token}([^A-Za-z0-9_]|$)"
            private_process_match
            "${header_content}"
        )
        if(private_process_match)
            message(FATAL_ERROR
                "Installed public header ${header} exposes private process token '${token}'"
            )
        endif()
    endforeach()
endforeach()

add_executable(runtime_installed_consumer main.cpp)
target_compile_features(runtime_installed_consumer PRIVATE cxx_std_20)
set_target_properties(runtime_installed_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(runtime_installed_consumer PRIVATE holoscan::core)

include(CTest)
add_test(NAME runtime_installed_consumer.run COMMAND runtime_installed_consumer)
if(WIN32)
    set_tests_properties(runtime_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/bin")
elseif(APPLE)
    set_tests_properties(runtime_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "DYLD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
else()
    set_tests_properties(runtime_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
endif()
