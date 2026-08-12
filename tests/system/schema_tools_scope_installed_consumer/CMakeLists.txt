# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

project(holoscan_schema_tools_scope_installed_consumer LANGUAGES CXX)

if(NOT DEFINED HOLOSCAN_TEST_INSTALL_PREFIX OR HOLOSCAN_TEST_INSTALL_PREFIX STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_TEST_INSTALL_PREFIX must name the isolated SDK install prefix")
endif()

function(find_holoscan)
    find_package(holoscan REQUIRED CONFIG
        PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}"
        NO_DEFAULT_PATH
    )
endfunction()

find_holoscan()

if(NOT TARGET holoscan::core)
    message(FATAL_ERROR "Installed package did not define holoscan::core")
endif()
if(NOT TARGET flatbuffers::flatc)
    message(FATAL_ERROR "Installed package did not provide flatbuffers::flatc")
endif()
if(NOT TARGET holoscan::fbs_tools)
    message(FATAL_ERROR "Installed package did not define holoscan::fbs_tools")
endif()

holoscan_add_flatbuffer_schema(
    TARGET scoped_message_generated
    SCHEMAS scoped_message.fbs
)

add_custom_target(schema_tools_scope_installed_consumer ALL)
add_dependencies(schema_tools_scope_installed_consumer scoped_message_generated)
