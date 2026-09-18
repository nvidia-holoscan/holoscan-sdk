# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_SDK_SOURCE_DIR
        HOLOSCAN_CONSUMER_SOURCE_DIR
        HOLOSCAN_CONSUMER_WORK_DIR
        HOLOSCAN_GENERATOR
        HOLOSCAN_CXX_COMPILER
        HOLOSCAN_CTEST_COMMAND
        HOLOSCAN_EXPECT_SHARED)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

if(HOLOSCAN_EXPECT_SHARED)
    set(package_lane shared)
else()
    set(package_lane static)
endif()
set(package_work_dir "${HOLOSCAN_CONSUMER_WORK_DIR}/${package_lane}")
set(install_prefix "${package_work_dir}/install")
file(REMOVE_RECURSE "${package_work_dir}")

# Build every exported target exercised by this consumer. The install emits import files for
# every export set (holoscan-targets.cmake, holoscan-visualization-targets.cmake), so an unbuilt
# exported target makes find_package fail.
set(sdk_build_command
    "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}"
    --target core sensor_io holoviz op_holoviz)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND sdk_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${sdk_build_command} RESULT_VARIABLE sdk_build_result)
if(NOT sdk_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build SDK targets required by the operator installed consumer")
endif()

# Component isolation: the module payload (both libraries, both header trees and the module's
# package files) must arrive from the holoscan-visualization component alone. The full install
# below also brings holoscan-dependencies, the SDK's default component, so an artifact that fell
# back to that default would otherwise go unnoticed here and in the per-component packages.
set(visualization_only_prefix "${package_work_dir}/visualization-only")
set(install_command
    "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
    --prefix "${visualization_only_prefix}"
    --component holoscan-visualization
)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND install_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${install_command} RESULT_VARIABLE install_result)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "Failed to install the holoscan-visualization component on its own")
endif()
if(WIN32)
    set(visualization_only_libraries
        "${visualization_only_prefix}/bin/holoviz.dll"
        "${visualization_only_prefix}/lib/holoviz.lib"
        "${visualization_only_prefix}/bin/holoscan_op_holoviz.dll"
        "${visualization_only_prefix}/lib/holoscan_op_holoviz.lib")
else()
    set(visualization_only_libraries
        "${visualization_only_prefix}/lib*/libholoviz.so"
        "${visualization_only_prefix}/lib*/libholoscan_op_holoviz.so")
endif()
foreach(required_glob IN LISTS visualization_only_libraries ITEMS
        "${visualization_only_prefix}/include/holoviz/holoviz.hpp"
        "${visualization_only_prefix}/include/holoviz/holoviz_export.h"
        "${visualization_only_prefix}/include/holoviz/imgui/vendor/imgui.h"
        "${visualization_only_prefix}/include/holoscan/visualization/holoviz_op.hpp"
        "${visualization_only_prefix}/lib*/cmake/holoscan/holoscan-visualization-config.cmake"
        "${visualization_only_prefix}/lib*/cmake/holoscan/holoscan-visualization-config-version.cmake"
        "${visualization_only_prefix}/lib*/cmake/holoscan/holoscan-visualization-targets.cmake")
    file(GLOB required_matches LIST_DIRECTORIES FALSE "${required_glob}")
    if(NOT required_matches)
        message(FATAL_ERROR
            "The holoscan-visualization component alone did not install ${required_glob}")
    endif()
endforeach()
file(GLOB_RECURSE foreign_libraries LIST_DIRECTORIES FALSE
    "${visualization_only_prefix}/*holoscan_core*")
if(foreign_libraries)
    message(FATAL_ERROR
        "The holoscan-visualization component installed Core artifacts: ${foreign_libraries}")
endif()
file(REMOVE_RECURSE "${visualization_only_prefix}")

foreach(component IN ITEMS
        holoscan
        holoscan-dependencies
        iceoryx2-lib
        dev
        holoscan-holomq
        holoscan-core
        holoscan-visualization
        holoscan-sensor-io)
    set(install_command
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${install_prefix}"
        --component "${component}"
    )
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND install_command --config "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(COMMAND ${install_command} RESULT_VARIABLE install_result)
    if(NOT install_result EQUAL 0)
        message(FATAL_ERROR "Failed to install SDK component ${component}")
    endif()
endforeach()

file(GLOB holoscan_package_dirs
    LIST_DIRECTORIES TRUE
    "${install_prefix}/*/cmake/holoscan"
    "${install_prefix}/*/*/cmake/holoscan")
if(NOT holoscan_package_dirs)
    message(FATAL_ERROR "Installed Holoscan package directory is missing under ${install_prefix}")
endif()
list(GET holoscan_package_dirs 0 holoscan_package_dir)
get_filename_component(install_library_dir "${holoscan_package_dir}/../.." ABSOLUTE)

foreach(required_install_path IN ITEMS
        "${install_prefix}/include/holoscan/holoscan.hpp"
        "${install_prefix}/include/holoscan/core/execution_context.hpp"
        "${install_prefix}/include/holoscan/core/compile.hpp"
        "${install_prefix}/include/holoscan/core/connection_options.hpp"
        "${install_prefix}/include/holoscan/core/external_topic.hpp"
        "${install_prefix}/include/holoscan/core/graph.hpp"
        "${install_prefix}/include/holoscan/core/operator.hpp"
        "${install_prefix}/include/holoscan/core/operator_spec.hpp"
        "${install_prefix}/include/holoscan/core/parameter_compute_control.hpp"
        "${install_prefix}/include/holoscan/core/parameter_control.hpp"
        "${install_prefix}/include/holoscan/core/parameter_snapshot.hpp"
        "${install_prefix}/include/holoscan/core/parameter_update.hpp"
        "${install_prefix}/include/holoscan/core/port.hpp"
        "${install_prefix}/include/holoscan/core/rate.hpp"
        "${install_prefix}/include/holoscan/core/rate_generation.hpp"
        "${install_prefix}/include/holoscan/core/readiness_source.hpp"
        "${install_prefix}/include/holoscan/core/run.hpp"
        "${install_prefix}/include/holoscan/core/run_session.hpp"
        "${install_prefix}/include/holoscan/core/predicate.hpp"
        "${install_prefix}/include/holoscan/core/payload_identity.hpp"
        "${install_prefix}/include/holoscan/core/payload_options.hpp"
        "${install_prefix}/include/holoscan/core/sample.hpp"
        "${install_prefix}/include/holoscan/core/schema_identity.hpp"
        "${install_prefix}/include/holoscan/core/temporal_contract.hpp"
        "${install_prefix}/include/holoscan/core/temporal_contract_records.hpp"
        "${install_prefix}/include/holoscan/core/tensor_contract.hpp"
        "${install_prefix}/include/holoscan/channel/manual_tensor_publisher.hpp"
        "${install_prefix}/include/holoscan/channel/clock_health.hpp"
        "${install_prefix}/include/holoscan/channel/pod_type.hpp"
        "${install_prefix}/include/holoscan/channel/runtime.hpp"
        "${holoscan_package_dir}/holoscan-config.cmake"
        "${holoscan_package_dir}/holoscan-config-version.cmake"
        "${holoscan_package_dir}/holoscan-visualization-config.cmake"
        "${holoscan_package_dir}/holoscan-visualization-config-version.cmake"
        "${holoscan_package_dir}/holoscan-visualization-targets.cmake")
    if(NOT EXISTS "${required_install_path}")
        message(FATAL_ERROR "Installed consumer prerequisite is missing: ${required_install_path}")
    endif()
endforeach()

set(installed_config
    "${holoscan_package_dir}/holoscan-config.cmake")
set(installed_config_version
    "${holoscan_package_dir}/holoscan-config-version.cmake")
file(READ "${installed_config}" installed_config_text)
file(READ "${installed_config_version}" installed_config_version_text)
foreach(installed_package_text IN ITEMS installed_config_text installed_config_version_text)
    if("${${installed_package_text}}" MATCHES "HOLOSCAN_RAW_BUILD_TREE_PACKAGE_UNSUPPORTED")
        message(FATAL_ERROR "Installed package entry reused the raw build-tree rejection stub")
    endif()
endforeach()
if(NOT installed_config_text MATCHES "holoscan-full-config\\.cmake")
    message(FATAL_ERROR "Installed config does not contain the component-aware package front door")
endif()
if(NOT installed_config_version_text MATCHES "set\\(PACKAGE_VERSION")
    message(FATAL_ERROR "Installed package version entry is not a generated compatibility file")
endif()

file(GLOB installed_core_libraries
    LIST_DIRECTORIES FALSE
    "${install_library_dir}/libholoscan_core.*"
    "${install_library_dir}/holoscan_core.*"
)
list(SORT installed_core_libraries)
if(installed_core_libraries)
    list(GET installed_core_libraries 0 installed_core_library)
else()
    set(installed_core_library "")
endif()
if(DEFINED HOLOSCAN_EXPECT_SHARED AND NOT "${installed_core_library}" STREQUAL "" AND NOT WIN32)
    if(HOLOSCAN_EXPECT_SHARED)
        if(installed_core_library MATCHES "\\.(a|lib)$")
            message(FATAL_ERROR "Shared installed-consumer lane installed a static Core library")
        endif()
    elseif(NOT installed_core_library MATCHES "\\.(a|lib)$")
        message(FATAL_ERROR "Static installed-consumer lane did not install a static Core library")
    endif()
endif()

function(assert_no_sdk_source_path_leak content context)
    set(scan_text "${content}")
    foreach(allowed_path IN ITEMS
            "${install_prefix}"
            "${package_work_dir}"
            "${HOLOSCAN_CONSUMER_SOURCE_DIR}")
        string(REPLACE "${allowed_path}" "<allowed-consumer-path>" scan_text "${scan_text}")
    endforeach()
    string(FIND "${scan_text}" "${HOLOSCAN_SDK_SOURCE_DIR}" leak_offset)
    if(NOT leak_offset EQUAL -1)
        message(FATAL_ERROR
            "${context} leaked SDK source path ${HOLOSCAN_SDK_SOURCE_DIR}:\n${scan_text}")
    endif()
endfunction()

function(assert_no_sdk_package_metadata_leak content context)
    foreach(forbidden_path IN ITEMS "${HOLOSCAN_BUILD_DIR}" "${HOLOSCAN_SDK_SOURCE_DIR}")
        string(FIND "${content}" "${forbidden_path}" leak_offset)
        if(NOT leak_offset EQUAL -1)
            message(FATAL_ERROR
                "${context} leaked SDK source/build path ${forbidden_path}:\n${content}")
        endif()
    endforeach()
endfunction()

file(GLOB_RECURSE installed_package_metadata
    LIST_DIRECTORIES FALSE
    "${holoscan_package_dir}/*.cmake")
if(NOT installed_package_metadata)
    message(FATAL_ERROR "Installed package metadata is missing under ${holoscan_package_dir}")
endif()
foreach(installed_package_metadata_file IN LISTS installed_package_metadata)
    file(READ "${installed_package_metadata_file}" installed_package_metadata_text)
    assert_no_sdk_package_metadata_leak(
        "${installed_package_metadata_text}" "${installed_package_metadata_file}")
endforeach()

set(consumer_build_dir "${package_work_dir}/build")
    file(REMOVE_RECURSE "${consumer_build_dir}")
    set(consumer_configure_options
        -G "${HOLOSCAN_GENERATOR}"
        "-DCMAKE_CXX_COMPILER:FILEPATH=${HOLOSCAN_CXX_COMPILER}"
    )
    if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
        list(APPEND consumer_configure_options -A "${HOLOSCAN_GENERATOR_PLATFORM}")
    endif()
    if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
        list(APPEND consumer_configure_options -T "${HOLOSCAN_GENERATOR_TOOLSET}")
    endif()
    if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
        list(APPEND consumer_configure_options "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
    endif()
    if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
        list(APPEND consumer_configure_options "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${HOLOSCAN_TOOLCHAIN_FILE}")
    endif()
    if(DEFINED HOLOSCAN_MSVC_RUNTIME_LIBRARY AND NOT HOLOSCAN_MSVC_RUNTIME_LIBRARY STREQUAL "")
        list(APPEND consumer_configure_options
            "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${HOLOSCAN_MSVC_RUNTIME_LIBRARY}")
    endif()
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND consumer_configure_options "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
    endif()

    set(configure_command
        "${CMAKE_COMMAND}"
        -S "${HOLOSCAN_CONSUMER_SOURCE_DIR}"
        -B "${consumer_build_dir}"
        ${consumer_configure_options}
        "-DCMAKE_PREFIX_PATH:PATH=${install_prefix}"
        "-DHOLOSCAN_TEST_INSTALL_PREFIX:PATH=${install_prefix}"
    )
    if(DEFINED HOLOSCAN_CONSUMER_FIND_MODE AND NOT HOLOSCAN_CONSUMER_FIND_MODE STREQUAL "")
        list(APPEND configure_command
            "-DHOLOSCAN_PACKAGE_FIND_MODE:STRING=${HOLOSCAN_CONSUMER_FIND_MODE}")
    endif()
    if(DEFINED HOLOSCAN_MODULE_PACKAGE_DIR AND NOT HOLOSCAN_MODULE_PACKAGE_DIR STREQUAL "")
        list(APPEND configure_command
            "-DHOLOSCAN_MODULE_PACKAGE_DIR:PATH=${install_prefix}/${HOLOSCAN_MODULE_PACKAGE_DIR}")
    endif()

    set(all_configure_output "")
    foreach(configure_pass RANGE 1 2)
        execute_process(
            COMMAND ${configure_command}
            RESULT_VARIABLE configure_result
            OUTPUT_VARIABLE configure_stdout
            ERROR_VARIABLE configure_stderr
        )
        string(APPEND all_configure_output "${configure_stdout}\n${configure_stderr}\n")
        if(NOT configure_result EQUAL 0)
            message(FATAL_ERROR
                "Failed to configure installed-package consumer on pass "
                "${configure_pass}:\n${configure_stdout}\n${configure_stderr}")
        endif()
    endforeach()
    if(NOT EXISTS "${consumer_build_dir}/holoscan-package-accepted")
        message(FATAL_ERROR "Installed-package consumer was not accepted")
    endif()
    assert_no_sdk_source_path_leak("${all_configure_output}" "configure output")
    foreach(configure_artifact IN ITEMS
            "${consumer_build_dir}/CMakeCache.txt"
            "${consumer_build_dir}/build.ninja")
        if(EXISTS "${configure_artifact}")
            file(READ "${configure_artifact}" configure_artifact_text)
            assert_no_sdk_source_path_leak(
                "${configure_artifact_text}" "${configure_artifact}")
        endif()
    endforeach()

    set(consumer_build_command "${CMAKE_COMMAND}" --build "${consumer_build_dir}" --verbose)
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND consumer_build_command --config "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(
        COMMAND ${consumer_build_command}
        RESULT_VARIABLE consumer_build_result
        OUTPUT_VARIABLE consumer_build_stdout
        ERROR_VARIABLE consumer_build_stderr
    )
    set(consumer_build_output "${consumer_build_stdout}\n${consumer_build_stderr}")
    if(NOT consumer_build_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to build installed-package consumer:\n${consumer_build_output}")
    endif()
    assert_no_sdk_source_path_leak("${consumer_build_output}" "verbose build output")
    file(GLOB_RECURSE consumer_link_metadata
        LIST_DIRECTORIES FALSE "${consumer_build_dir}/*link.txt")
    foreach(link_metadata IN LISTS consumer_link_metadata)
        file(READ "${link_metadata}" link_metadata_text)
        assert_no_sdk_source_path_leak("${link_metadata_text}" "link metadata")
    endforeach()

    set(consumer_test_command
        "${HOLOSCAN_CTEST_COMMAND}" --test-dir "${consumer_build_dir}" --output-on-failure
    )
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND consumer_test_command -C "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(
        COMMAND ${consumer_test_command}
        RESULT_VARIABLE consumer_test_result
        OUTPUT_VARIABLE consumer_test_stdout
        ERROR_VARIABLE consumer_test_stderr
    )
    set(consumer_test_output "${consumer_test_stdout}\n${consumer_test_stderr}")
    assert_no_sdk_source_path_leak("${consumer_test_output}" "ctest output")
    if(NOT consumer_test_result EQUAL 0)
        message(FATAL_ERROR
            "Installed-package consumer execution failed:\n"
            "${consumer_test_output}")
    endif()

# Use the same installed SDK to check direct HoloViz consumption, after moving it so no original
# install path can accidentally satisfy dependency discovery or shared-library loading.
file(RENAME "${install_prefix}" "${package_work_dir}/relocated")
set(install_prefix "${package_work_dir}/relocated")
set(HOLOVIZ_CONSUMER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/holoviz_installed_consumer")
include("${CMAKE_CURRENT_LIST_DIR}/run_holoviz_installed_consumer.cmake")
