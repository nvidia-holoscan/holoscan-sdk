# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoviz_installed_consumer LANGUAGES CXX)

set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY TRUE)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY TRUE)

# An external visualization operator reaches HoloViz through the SDK package. "component" is the
# documented form and is repeated to prove the loader is re-entrant; "unqualified" checks that an
# aggregate find_package(holoscan) also exposes the packaged module.
if(HOLOVIZ_DISCOVERY_MODE STREQUAL "component")
    find_package(holoscan 5 COMPONENTS visualization
        PATHS ${CMAKE_PREFIX_PATH} NO_DEFAULT_PATH REQUIRED)
    find_package(holoscan 5 COMPONENTS visualization
        PATHS ${CMAKE_PREFIX_PATH} NO_DEFAULT_PATH REQUIRED)
    if(NOT holoscan_visualization_FOUND)
        message(FATAL_ERROR "The visualization component was not reported as found")
    endif()
elseif(HOLOVIZ_DISCOVERY_MODE STREQUAL "unqualified")
    find_package(holoscan 5 PATHS ${CMAKE_PREFIX_PATH} NO_DEFAULT_PATH REQUIRED)
else()
    message(FATAL_ERROR "Unknown HOLOVIZ_DISCOVERY_MODE: ${HOLOVIZ_DISCOVERY_MODE}")
endif()

# The visualization component ships the build-time shader compilation helper beside
# holoscan-config.cmake, so a downstream project can generate the same embedded-SPIR-V headers the
# library uses. The helper needs a slangc only when it is called, so its presence is what is
# asserted here.
if(NOT COMMAND holoviz_generate_shader_header)
    message(FATAL_ERROR
        "The installed SDK package did not provide holoviz_generate_shader_header()")
endif()

foreach(visualization_target IN ITEMS
        holoscan::visualization::holoviz holoscan::visualization::op_holoviz)
    if(NOT TARGET ${visualization_target})
        message(FATAL_ERROR "The SDK package did not import ${visualization_target}")
    endif()
endforeach()

get_target_property(holoviz_type holoscan::visualization::holoviz TYPE)
get_target_property(holoviz_imported holoscan::visualization::holoviz IMPORTED)
if(NOT holoviz_type STREQUAL "SHARED_LIBRARY" OR NOT holoviz_imported)
    message(FATAL_ERROR "Expected an imported shared holoscan::visualization::holoviz target")
endif()
get_target_property(holoviz_configurations holoscan::visualization::holoviz IMPORTED_CONFIGURATIONS)
foreach(configuration IN LISTS holoviz_configurations)
    get_target_property(private_dependencies holoscan::visualization::holoviz
        IMPORTED_LINK_DEPENDENT_LIBRARIES_${configuration})
    if(private_dependencies)
        message(FATAL_ERROR "HoloViz exposed private runtime targets: ${private_dependencies}")
    endif()
endforeach()
get_target_property(holoviz_links holoscan::visualization::holoviz INTERFACE_LINK_LIBRARIES)
foreach(dependency IN ITEMS CUDA::cuda_driver tl::expected)
    if(NOT dependency IN_LIST holoviz_links OR NOT TARGET ${dependency})
        message(FATAL_ERROR "Missing public HoloViz dependency: ${dependency}")
    endif()
endforeach()
foreach(dependency IN LISTS holoviz_links)
    if(NOT dependency STREQUAL "CUDA::cuda_driver" AND NOT dependency STREQUAL "tl::expected")
        message(FATAL_ERROR "Unexpected HoloViz public dependency: ${dependency}")
    endif()
endforeach()
get_target_property(holoviz_features holoscan::visualization::holoviz INTERFACE_COMPILE_FEATURES)
if(NOT "cxx_std_20" IN_LIST holoviz_features)
    message(FATAL_ERROR "HoloViz did not propagate C++20")
endif()
# HoloViz is an implementation detail of HolovizOp: its installed headers include no <holoviz/**>
# header, so consumers of holoscan::visualization::op_holoviz must not inherit
# holoscan::visualization::holoviz as a usage requirement. It may still appear as a private
# runtime dependency of the shared library.
get_target_property(operator_links holoscan::visualization::op_holoviz INTERFACE_LINK_LIBRARIES)
if("holoscan::visualization::holoviz" IN_LIST operator_links)
    message(FATAL_ERROR "holoscan::visualization::op_holoviz leaks "
        "holoscan::visualization::holoviz into its public interface")
endif()

add_executable(holoviz_installed_consumer test_holoviz_consumer.cpp)
target_link_libraries(holoviz_installed_consumer PRIVATE holoscan::visualization::holoviz)

include(CTest)
add_test(NAME holoviz_installed_consumer.run COMMAND holoviz_installed_consumer)
if(WIN32)
    set_tests_properties(holoviz_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:holoscan::visualization::holoviz>")
endif()
