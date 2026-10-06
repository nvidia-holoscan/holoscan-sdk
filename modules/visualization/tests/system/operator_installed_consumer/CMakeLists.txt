# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_operator_installed_consumer LANGUAGES CXX)

set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY TRUE)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY TRUE)
if(HOLOSCAN_PACKAGE_FIND_MODE STREQUAL "module")
    if(NOT DEFINED HOLOSCAN_MODULE_PACKAGE_DIR OR HOLOSCAN_MODULE_PACKAGE_DIR STREQUAL "")
        message(FATAL_ERROR "HOLOSCAN_MODULE_PACKAGE_DIR is required for module package discovery")
    endif()
    set(holoscan-visualization_DIR "${HOLOSCAN_MODULE_PACKAGE_DIR}")
    find_package(holoscan-visualization CONFIG REQUIRED)
else()
    find_package(holoscan COMPONENTS visualization REQUIRED
        PATHS "${CMAKE_PREFIX_PATH}"
        NO_DEFAULT_PATH
    )
endif()

if(NOT TARGET holoscan::core)
    message(FATAL_ERROR "Installed package did not define holoscan::core")
endif()
if(HOLOSCAN_PACKAGE_FIND_MODE STREQUAL "module")
    if(NOT holoscan-visualization_FOUND)
        message(FATAL_ERROR "Installed module package did not report itself as found")
    endif()
elseif(NOT holoscan_visualization_FOUND)
    message(FATAL_ERROR "Installed package did not report the visualization component")
endif()
# The visualization component imports HolovizOp together with the HoloViz library it links.
foreach(visualization_target IN ITEMS
        holoscan::visualization::op_holoviz holoscan::visualization::holoviz)
    if(NOT TARGET ${visualization_target})
        message(FATAL_ERROR "Installed package did not define ${visualization_target}")
    endif()
endforeach()
# Both discovery paths load the build-time shader compilation helper with the component.
if(NOT COMMAND holoviz_generate_shader_header)
    message(FATAL_ERROR "Installed package did not provide holoviz_generate_shader_header()")
endif()
# The module owns the holoscan::visualization:: namespace; the flat names must not come back.
foreach(legacy_target IN ITEMS holoscan::holoviz holoscan::op_holoviz holoviz::holoviz)
    if(TARGET ${legacy_target})
        message(FATAL_ERROR "Installed package defined the retired target name ${legacy_target}")
    endif()
endforeach()
if(HOLOSCAN_PACKAGE_FIND_MODE STREQUAL "module")
    get_filename_component(HOLOSCAN_TEST_INSTALL_PREFIX "${holoscan-visualization_DIR}/../../.." ABSOLUTE)
    get_filename_component(HOLOSCAN_TEST_LIBRARY_DIR "${holoscan-visualization_DIR}/../.." ABSOLUTE)
    file(WRITE "${CMAKE_BINARY_DIR}/holoscan-package-accepted" "${holoscan-visualization_DIR}\n")
else()
    get_filename_component(HOLOSCAN_TEST_INSTALL_PREFIX "${holoscan_DIR}/../../.." ABSOLUTE)
    get_filename_component(HOLOSCAN_TEST_LIBRARY_DIR "${holoscan_DIR}/../.." ABSOLUTE)
    file(WRITE "${CMAKE_BINARY_DIR}/holoscan-package-accepted" "${holoscan_DIR}\n")
endif()
# The layer renderer is compiled into the operator DSO; none of its private headers may install.
foreach(private_header IN ITEMS holoviz/frame_data.hpp holoviz/render_backend.hpp
        holoviz_op/backends/render_backend.hpp holoviz_op/backends/holoviz_render_backend.hpp)
    if(EXISTS "${HOLOSCAN_TEST_INSTALL_PREFIX}/include/${private_header}")
        message(FATAL_ERROR "Installed package exposed private backend header ${private_header}")
    endif()
endforeach()
# HolovizOp's installed API is renderer-free: no public header may include a HoloViz header.
file(GLOB_RECURSE operator_public_headers
    "${HOLOSCAN_TEST_INSTALL_PREFIX}/include/holoscan/visualization/*.hpp")
if(NOT operator_public_headers)
    message(FATAL_ERROR "Installed package did not ship <holoscan/visualization/*.hpp>")
endif()
foreach(public_header IN LISTS operator_public_headers)
    file(STRINGS "${public_header}" holoviz_includes REGEX "#include +[<\"]holoviz/")
    if(holoviz_includes)
        message(FATAL_ERROR
            "${public_header} includes a HoloViz header: ${holoviz_includes}")
    endif()
endforeach()

add_executable(operator_installed_consumer
    main.cpp
    holoviz_operator_smoke.cpp
    parameter_compute_header_smoke.cpp
    runtime_header_smoke.cpp
)
target_compile_features(operator_installed_consumer PRIVATE cxx_std_20)
set_target_properties(operator_installed_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(operator_installed_consumer PRIVATE
    holoscan::core
    holoscan::visualization::op_holoviz
)

include(CTest)
add_test(NAME operator_installed_consumer.run COMMAND operator_installed_consumer)
if(WIN32)
    set_tests_properties(operator_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/bin")
elseif(APPLE)
    set_tests_properties(operator_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "DYLD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_LIBRARY_DIR}")
else()
    set_tests_properties(operator_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_LIBRARY_DIR}")
endif()
