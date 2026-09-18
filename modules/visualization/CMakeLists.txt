# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# In this directory:
# - Holoscan Visualization: Top-level project bundling HolovizOp and Holoviz
# - Holoviz: Backend library for real-time GPU visualization, agnostic of Holoscan SDK APIs
# - HolovizOp: Holoscan SDK operator wrapping Holoviz for pipeline integration
#
# To build:
# - Default: Build the top-level Holoscan SDK with "-DHOLOSCAN_BUILD_VISUALIZATION=ON"
# - (Alternative) standalone: Build from this source and provide a standalone Holoscan SDK installation
#
# To consume: find_package(holoscan COMPONENTS visualization) imports
# holoscan::visualization::holoviz (the rendering library) and holoscan::visualization::op_holoviz
# (HolovizOp). Every target the module ships lives in the holoscan::visualization:: namespace.
# Both ship in the module's own export set (holoscan-visualization-exports ->
# holoscan-visualization-targets.cmake), reachable either through that component or directly via
# find_package(holoscan-visualization).

cmake_minimum_required(VERSION 4.0)

if(NOT holoscan_VERSION)
    set(holoscan_VERSION 5.0.0)
endif()
project(holoscan_visualization
    VERSION ${holoscan_VERSION}
    DESCRIPTION "Holoscan Visualization module"
    LANGUAGES CXX
)

include(GenerateExportHeader)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT holoscan-visualization)
set(HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT holoscan-visualization-examples)

if(holoscan_visualization_IS_TOP_LEVEL)
    # Allow building Holoscan Visualization as a standalone project against a pre-existing Holoscan SDK core
    option(HOLOSCAN_BUILD_EXAMPLES "Build HoloViz examples" OFF)
    option(HOLOSCAN_BUILD_TESTS "Build HoloViz tests" ON)

    # Request Core explicitly: an unqualified find_package(holoscan) also imports every optional
    # component the SDK prefix carries, and an SDK installed with the visualization module would
    # then define the very holoscan::visualization::holoviz / holoscan::visualization::op_holoviz
    # targets this project builds.
    find_package(holoscan 5 REQUIRED COMPONENTS core)

    include(SetupRapidsCMake OPTIONAL)
    include(setup_dependencies OPTIONAL)
    # Shared-library hardening helper from the SDK source tree; absent from the installed package.
    include(HoloscanTargetConfiguration OPTIONAL)

    # Check for Holoscan CMake modules existence
    if(NOT COMMAND rapids_cpm_find OR NOT COMMAND holoscan_add_dependency)
        message(FATAL_ERROR "Failed to load Holoscan CMake dependency functions. Please add to CMAKE_MODULE_PATH and try again.")
    endif()
    if(HOLOSCAN_BUILD_EXAMPLES)
        # The example helpers ship with the installed SDK next to holoscan-config.cmake and also
        # live in the SDK source tree under public/examples/cmake.
        include(HoloscanExampleInstall OPTIONAL)
        if(NOT COMMAND holoscan_configure_example_install
           AND EXISTS "${holoscan_DIR}/HoloscanExampleInstall.cmake")
            include("${holoscan_DIR}/HoloscanExampleInstall.cmake")
        endif()
        if(NOT COMMAND holoscan_configure_example_install)
            message(FATAL_ERROR "Failed to load Holoscan CMake example functions (HoloscanExampleInstall.cmake). Please add to CMAKE_MODULE_PATH and try again.")
        endif()
    endif()
endif()

include(CTest)

include(holoviz_dependencies)

##########################################################
# Holoviz backend library
##########################################################
add_subdirectory(src)

##########################################################
# HolovizOp
##########################################################
add_subdirectory(operators)

##########################################################
# Holoscan Visualization
##########################################################
if(HOLOSCAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(tests)
endif()

##########################################################
# Installation (based on project config)
##########################################################
set(_holoscan_visualization_package_dir
    "${CMAKE_CURRENT_BINARY_DIR}/holoscan-visualization-package")
set(_holoscan_visualization_cmake_install_dir
    "${CMAKE_INSTALL_LIBDIR}/cmake/holoscan")
file(MAKE_DIRECTORY "${_holoscan_visualization_package_dir}")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
    "${_holoscan_visualization_package_dir}/holoscan-visualization-config.cmake"
    INSTALL_DESTINATION "${_holoscan_visualization_cmake_install_dir}"
)
write_basic_package_version_file(
    "${_holoscan_visualization_package_dir}/holoscan-visualization-config-version.cmake"
    COMPATIBILITY SameMajorVersion
)

install(EXPORT holoscan-visualization-exports
    FILE holoscan-visualization-targets.cmake
    NAMESPACE holoscan::
    DESTINATION "${_holoscan_visualization_cmake_install_dir}"
    COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
)
install(FILES
    "${_holoscan_visualization_package_dir}/holoscan-visualization-config.cmake"
    "${_holoscan_visualization_package_dir}/holoscan-visualization-config-version.cmake"
    DESTINATION "${_holoscan_visualization_cmake_install_dir}"
    COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
)
if(holoscan_visualization_IS_TOP_LEVEL)
    # Top-level module packages ship the canonical SDK license and their notice at the package root.
    # When built as part of the Holoscan SDK, the SDK installs these files centrally.
    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/../../LICENSE.txt"
        NOTICE.txt
        DESTINATION "${CMAKE_INSTALL_DOCDIR}"
        COMPONENT ${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}
    )
endif()

unset(_holoscan_visualization_cmake_install_dir)
unset(_holoscan_visualization_package_dir)
