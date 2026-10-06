# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_installed_external_clients_source LANGUAGES NONE)

include(CTest)

if(NOT DEFINED EXAMPLE_SOURCE OR EXAMPLE_SOURCE STREQUAL "")
    message(FATAL_ERROR "EXAMPLE_SOURCE is required")
endif()

set(HOLOSCAN_BUILD_PYTHON_EXAMPLES ON)
set(HOLOSCAN_EXAMPLES_IN_SDK_BUILD OFF)
set(HOLOSCAN_INSTALL_EXAMPLE_SOURCE OFF)
set(HOLOSCAN_EXAMPLES_INSTALL_DIR examples)
set(HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR unused)

# Limit this fixture to source-inventory generation. Native build and installation behavior is
# covered by the enclosing installed-examples package test.
function(add_executable target)
    _add_executable(${target} IMPORTED GLOBAL)
    set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${CMAKE_COMMAND}")
endfunction()

function(target_compile_features)
endfunction()

function(target_link_libraries)
endfunction()

function(holoscan_configure_example_install)
endfunction()

function(install)
endfunction()

add_subdirectory("${EXAMPLE_SOURCE}" external_clients)
