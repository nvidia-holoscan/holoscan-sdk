# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Global dependency helper definitions only. Feature-specific setup files decide
# which dependency modules to load and which temporary CMake state to adjust.

include(GNUInstallDirs)
function(holoscan_add_dependency module_name)
    cmake_parse_arguments(PARSE_ARGV 1 _HOLOSCAN_ADD_DEPENDENCY "" "COMPONENT;EXPORT_SET" "")
    if(NOT "${_HOLOSCAN_ADD_DEPENDENCY_UNPARSED_ARGUMENTS}" STREQUAL "")
        message(FATAL_ERROR
            "Received unknown argument(s): ${_HOLOSCAN_ADD_DEPENDENCY_UNPARSED_ARGUMENTS}")
    endif()

    # Ensure the dependency export always has some nontrivial component and export set
    if(_HOLOSCAN_ADD_DEPENDENCY_COMPONENT)
        set(HOLOSCAN_DEPEND_INSTALL_COMPONENT "${_HOLOSCAN_ADD_DEPENDENCY_COMPONENT}")
    else()
        set(HOLOSCAN_DEPEND_INSTALL_COMPONENT "holoscan-dependencies")
    endif()
    if(_HOLOSCAN_ADD_DEPENDENCY_EXPORT_SET)
        set(HOLOSCAN_DEPEND_EXPORT_SET "${_HOLOSCAN_ADD_DEPENDENCY_EXPORT_SET}")
    else()
        set(HOLOSCAN_DEPEND_EXPORT_SET "holoscan-exports")
    endif()

    set(_holoscan_dependency_file "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/deps/${module_name}.cmake")
    if(NOT EXISTS "${_holoscan_dependency_file}")
        message(FATAL_ERROR "Unknown dependency module '${module_name}': ${_holoscan_dependency_file}")
    endif()

    # Set configuration state for FetchContent initialization
    message(DEBUG "Adding dependent module '${module_name}' to install component '${HOLOSCAN_DEPEND_INSTALL_COMPONENT}'")
    set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "${HOLOSCAN_DEPEND_INSTALL_COMPONENT}")
    # Keep clang-tidy focused on Holoscan code, not third-party dependency targets.
    set(_HOLOSCAN_DEPEND_SAVED_CMAKE_CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY}")
    set(CMAKE_CXX_CLANG_TIDY "")

    include("${_holoscan_dependency_file}")

    set(CMAKE_CXX_CLANG_TIDY "${_HOLOSCAN_DEPEND_SAVED_CMAKE_CXX_CLANG_TIDY}")
    unset(_HOLOSCAN_DEPEND_SAVED_CMAKE_CXX_CLANG_TIDY)
    unset(HOLOSCAN_DEPEND_EXPORT_SET)
    unset(HOLOSCAN_DEPEND_INSTALL_COMPONENT)
endfunction()
