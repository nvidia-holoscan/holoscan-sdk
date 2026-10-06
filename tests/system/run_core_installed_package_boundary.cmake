# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_BOUNDARY_WORK_DIR
        HOLOSCAN_INSTALL_LIB_DIR
        HOLOSCAN_EXPECT_SHARED)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

set(install_prefix "${HOLOSCAN_BOUNDARY_WORK_DIR}/install")
file(REMOVE_RECURSE "${HOLOSCAN_BOUNDARY_WORK_DIR}")

# The installed package exports the complete Core target set. Build schema and Sensor I/O with Core
# so the isolated installation does not reference missing target artifacts during package loading.
set(sdk_build_targets core holoscan_schema sensor_io config config_io params params_io)
if(NOT HOLOSCAN_EXPECT_SHARED)
    list(APPEND sdk_build_targets spdlog)
endif()
set(sdk_build_command
    "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}" --target ${sdk_build_targets} --parallel 4)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND sdk_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${sdk_build_command} RESULT_VARIABLE sdk_build_result)
if(NOT sdk_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build Core targets for the installed-package boundary test")
endif()

foreach(component IN ITEMS
        holoscan
        holoscan-dependencies
        iceoryx2-lib
        dev
        holoscan-holomq
        holoscan-core
        holoscan-sensor-io
        holoscan-config
        holoscan-config-io)
    set(install_command
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${install_prefix}"
        --component "${component}")
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND install_command --config "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(COMMAND ${install_command} RESULT_VARIABLE install_result)
    if(NOT install_result EQUAL 0)
        message(FATAL_ERROR "Failed to install SDK component ${component}")
    endif()
endforeach()

set(installed_include_dir "${install_prefix}/include")
set(installed_core_include_dir "${installed_include_dir}/holoscan")
set(installed_package_dir "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan")
foreach(required_install_path IN ITEMS
        "${installed_include_dir}/holoscan/holoscan.hpp"
        "${installed_core_include_dir}/core/graph.hpp"
        "${installed_package_dir}/holoscan-config.cmake")
    if(NOT EXISTS "${required_install_path}")
        message(FATAL_ERROR "Installed Core boundary prerequisite is missing: ${required_install_path}")
    endif()
endforeach()

file(GLOB installed_core_libraries
    LIST_DIRECTORIES FALSE
    "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/libholoscan_core.*"
    "${install_prefix}/${HOLOSCAN_INSTALL_LIB_DIR}/holoscan_core.*")
list(SORT installed_core_libraries)
if(NOT installed_core_libraries)
    message(FATAL_ERROR "Installed Core boundary prerequisite is missing: holoscan_core library")
endif()
list(GET installed_core_libraries 0 installed_core_library)

if(NOT WIN32)
    if(HOLOSCAN_EXPECT_SHARED AND installed_core_library MATCHES "\\.(a|lib)$")
        message(FATAL_ERROR "Shared boundary test installed a static Core library")
    elseif(NOT HOLOSCAN_EXPECT_SHARED AND NOT installed_core_library MATCHES "\\.(a|lib)$")
        message(FATAL_ERROR "Static boundary test did not install a static Core library")
    endif()
endif()

function(run_boundary_check check_name)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE check_result)
    if(NOT check_result EQUAL 0)
        message(FATAL_ERROR "Installed Core boundary check failed: ${check_name}")
    endif()
endfunction()

run_boundary_check(operator_public_surface
    "${CMAKE_COMMAND}"
    "-DHOLOSCAN_PUBLIC_INCLUDE_DIR:PATH=${installed_include_dir}"
    -P "${CMAKE_CURRENT_LIST_DIR}/verify_operator_public_surface.cmake")

run_boundary_check(graph_public_surface
    "${CMAKE_COMMAND}"
    "-DHOLOSCAN_PUBLIC_INCLUDE_DIR:PATH=${installed_core_include_dir}"
    "-DHOLOSCAN_PACKAGE_ROOT:PATH=${install_prefix}"
    -P "${CMAKE_CURRENT_LIST_DIR}/../core/check_graph_public_surface.cmake")

run_boundary_check(graph_exported_symbols
    "${CMAKE_COMMAND}"
    "-DHOLOSCAN_CORE_LIBRARY:FILEPATH=${installed_core_library}"
    "-DHOLOSCAN_NM_EXECUTABLE:FILEPATH=${HOLOSCAN_NM_EXECUTABLE}"
    "-DHOLOSCAN_READELF_EXECUTABLE:FILEPATH=${HOLOSCAN_READELF_EXECUTABLE}"
    "-DHOLOSCAN_DUMPBIN_EXECUTABLE:FILEPATH=${HOLOSCAN_DUMPBIN_EXECUTABLE}"
    -P "${CMAKE_CURRENT_LIST_DIR}/../core/check_graph_exported_symbols.cmake")
