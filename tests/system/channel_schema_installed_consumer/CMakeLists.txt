# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

project(holoscan_channel_schema_installed_consumer LANGUAGES CXX)

# REQ: CHAN-074
# REQ: CHAN-075
# REQ: CHAN-120
# REQ: CHAN-077

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
if(NOT TARGET holoscan::schema)
    message(FATAL_ERROR "Installed package did not define holoscan::schema")
endif()

add_executable(channel_schema_installed_consumer ../holomq_schema_smoke.cpp)
target_compile_features(channel_schema_installed_consumer PRIVATE cxx_std_20)
set_target_properties(channel_schema_installed_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(channel_schema_installed_consumer PRIVATE holoscan::core holoscan::schema)

include(CTest)
add_test(NAME channel_schema_installed_consumer.run COMMAND channel_schema_installed_consumer)
if(WIN32)
    set_tests_properties(channel_schema_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/bin")
elseif(APPLE)
    set_tests_properties(channel_schema_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "DYLD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
else()
    set_tests_properties(channel_schema_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
endif()
