# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_BUILD_DIR
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

set(sdk_build_command "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}" --target core)
if(DEFINED HOLOSCAN_BUILD_CONFIG AND NOT HOLOSCAN_BUILD_CONFIG STREQUAL "")
    list(APPEND sdk_build_command --config "${HOLOSCAN_BUILD_CONFIG}")
endif()
execute_process(COMMAND ${sdk_build_command} RESULT_VARIABLE sdk_build_result)
if(NOT sdk_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build SDK targets required by the operator installed consumer")
endif()

foreach(component IN ITEMS
        holoscan
        holoscan-dependencies
        holomq-iceoryx2-libraries
        dev
        holomq
        holoscan-core)
    set(install_command
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${install_prefix}"
        --component "${component}"
    )
    if(DEFINED HOLOSCAN_BUILD_CONFIG AND NOT HOLOSCAN_BUILD_CONFIG STREQUAL "")
        list(APPEND install_command --config "${HOLOSCAN_BUILD_CONFIG}")
    endif()
    execute_process(COMMAND ${install_command} RESULT_VARIABLE install_result)
    if(NOT install_result EQUAL 0)
        message(FATAL_ERROR "Failed to install SDK component ${component}")
    endif()
endforeach()

foreach(required_install_path IN ITEMS
        "${install_prefix}/include/holoscan/holoscan.hpp"
        "${install_prefix}/include/holoscan/core/execution_context.hpp"
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
        "${install_prefix}/include/holoscan/channel/manual_tensor_publisher.hpp"
        "${install_prefix}/include/holoscan/channel/clock_health.hpp"
        "${install_prefix}/include/holoscan/channel/pod_type.hpp"
        "${install_prefix}/include/holoscan/channel/runtime.hpp"
        "${install_prefix}/lib/cmake/holoscan/holoscan-config.cmake"
        "${install_prefix}/lib/cmake/holoscan/holoscan-config-version.cmake")
    if(NOT EXISTS "${required_install_path}")
        message(FATAL_ERROR "Installed consumer prerequisite is missing: ${required_install_path}")
    endif()
endforeach()

set(installed_config
    "${install_prefix}/lib/cmake/holoscan/holoscan-config.cmake")
set(installed_config_version
    "${install_prefix}/lib/cmake/holoscan/holoscan-config-version.cmake")
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
    "${install_prefix}/lib/libholoscan_core.*"
    "${install_prefix}/lib/holoscan_core.*"
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

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        "-DHOLOSCAN_PUBLIC_INCLUDE_DIR:PATH=${install_prefix}/include"
        -P "${CMAKE_CURRENT_LIST_DIR}/verify_operator_public_surface.cmake"
    RESULT_VARIABLE installed_surface_result
)
if(NOT installed_surface_result EQUAL 0)
    message(FATAL_ERROR "Installed Operator/Channel public-surface boundary validation failed")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        "-DHOLOSCAN_PUBLIC_INCLUDE_DIR:PATH=${install_prefix}/include/holoscan"
        "-DHOLOSCAN_PACKAGE_ROOT:PATH=${install_prefix}"
        -P "${CMAKE_CURRENT_LIST_DIR}/../core/check_graph_public_surface.cmake"
    RESULT_VARIABLE installed_graph_surface_result
)
if(NOT installed_graph_surface_result EQUAL 0)
    message(FATAL_ERROR "Installed Graph public-surface boundary validation failed")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}"
        "-DHOLOSCAN_NM_EXECUTABLE:FILEPATH=${HOLOSCAN_NM_EXECUTABLE}"
        "-DHOLOSCAN_DUMPBIN_EXECUTABLE:FILEPATH=${HOLOSCAN_DUMPBIN_EXECUTABLE}"
        "-DHOLOSCAN_CORE_LIBRARY:FILEPATH=${installed_core_library}"
        -P "${CMAKE_CURRENT_LIST_DIR}/../core/check_graph_exported_symbols.cmake"
    RESULT_VARIABLE installed_graph_symbol_result
)
if(NOT installed_graph_symbol_result EQUAL 0)
    message(FATAL_ERROR "Installed Graph exported-symbol boundary validation failed")
endif()

get_filename_component(
    sdk_public_source_dir "${HOLOSCAN_CONSUMER_SOURCE_DIR}/../../.." ABSOLUTE)

function(assert_no_sdk_path_leak content context)
    set(scan_text "${content}")
    foreach(allowed_path IN ITEMS
            "${install_prefix}"
            "${package_work_dir}"
            "${HOLOSCAN_CONSUMER_SOURCE_DIR}")
        string(REPLACE "${allowed_path}" "<allowed-consumer-path>" scan_text "${scan_text}")
    endforeach()
    foreach(forbidden_path IN ITEMS "${HOLOSCAN_BUILD_DIR}" "${sdk_public_source_dir}")
        string(FIND "${scan_text}" "${forbidden_path}" leak_offset)
        if(NOT leak_offset EQUAL -1)
            message(FATAL_ERROR
                "${context} leaked SDK source/build path ${forbidden_path}:\n${scan_text}")
        endif()
    endforeach()
endfunction()

if(HOLOSCAN_PACKAGE_DISCOVERY_MATRIX)
    set(package_find_modes default core_component missing_component)
else()
    set(package_find_modes default)
endif()

foreach(find_mode IN LISTS package_find_modes)
    set(consumer_build_dir "${package_work_dir}/${find_mode}/build")
    file(REMOVE_RECURSE "${consumer_build_dir}")
    set(configure_command
        "${CMAKE_COMMAND}"
        -S "${HOLOSCAN_CONSUMER_SOURCE_DIR}"
        -B "${consumer_build_dir}"
        -G "${HOLOSCAN_GENERATOR}"
        "-DCMAKE_PREFIX_PATH:PATH=${install_prefix}"
        "-DHOLOSCAN_TEST_INSTALL_PREFIX:PATH=${install_prefix}"
        "-DHOLOSCAN_FIND_MODE:STRING=${find_mode}"
        "-DCMAKE_CXX_COMPILER:FILEPATH=${HOLOSCAN_CXX_COMPILER}"
    )
    if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
        list(APPEND configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
    endif()
    if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
        list(APPEND configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
    endif()
    if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
        list(APPEND configure_command "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
    endif()
    if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
        list(APPEND configure_command "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${HOLOSCAN_TOOLCHAIN_FILE}")
    endif()
    if(DEFINED HOLOSCAN_MSVC_RUNTIME_LIBRARY AND NOT HOLOSCAN_MSVC_RUNTIME_LIBRARY STREQUAL "")
        list(APPEND configure_command
            "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${HOLOSCAN_MSVC_RUNTIME_LIBRARY}")
    endif()
    if(DEFINED HOLOSCAN_BUILD_CONFIG AND NOT HOLOSCAN_BUILD_CONFIG STREQUAL "")
        list(APPEND configure_command "-DCMAKE_BUILD_TYPE:STRING=${HOLOSCAN_BUILD_CONFIG}")
    endif()

    if(find_mode STREQUAL "missing_component")
        execute_process(
            COMMAND ${configure_command}
            RESULT_VARIABLE configure_result
            OUTPUT_VARIABLE configure_stdout
            ERROR_VARIABLE configure_stderr
        )
        set(configure_output "${configure_stdout}\n${configure_stderr}")
        if(configure_result EQUAL 0)
            message(FATAL_ERROR "Unknown installed-package component unexpectedly configured")
        endif()
        if(configure_output MATCHES "HOLOSCAN_RAW_BUILD_TREE_PACKAGE_UNSUPPORTED")
            message(FATAL_ERROR "Installed component failure used the raw build-tree rejection")
        endif()
        # This fixture intentionally matches CMake's package-not-found diagnostic family to
        # distinguish a normal missing component from the SDK's raw-build-tree rejection. The
        # wording is not an SDK contract and may need adjustment when the supported CMake changes.
        if(NOT configure_output MATCHES
                "holoscan_FOUND to FALSE|package \"holoscan\" is considered to be NOT FOUND")
            message(FATAL_ERROR
                "Installed component failure did not use normal package-component semantics:\n"
                "${configure_output}")
        endif()
        continue()
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
                "Failed to configure ${find_mode} installed-package consumer on pass "
                "${configure_pass}:\n${configure_stdout}\n${configure_stderr}")
        endif()
    endforeach()
    if(HOLOSCAN_PACKAGE_DISCOVERY_MATRIX
            AND NOT EXISTS "${consumer_build_dir}/holoscan-package-accepted")
        message(FATAL_ERROR "${find_mode} installed-package consumer was not accepted")
    endif()
    assert_no_sdk_path_leak("${all_configure_output}" "${find_mode} configure output")
    foreach(configure_artifact IN ITEMS
            "${consumer_build_dir}/CMakeCache.txt"
            "${consumer_build_dir}/build.ninja")
        if(EXISTS "${configure_artifact}")
            file(READ "${configure_artifact}" configure_artifact_text)
            assert_no_sdk_path_leak(
                "${configure_artifact_text}" "${find_mode} ${configure_artifact}")
        endif()
    endforeach()

    set(consumer_build_command "${CMAKE_COMMAND}" --build "${consumer_build_dir}" --verbose)
    if(DEFINED HOLOSCAN_BUILD_CONFIG AND NOT HOLOSCAN_BUILD_CONFIG STREQUAL "")
        list(APPEND consumer_build_command --config "${HOLOSCAN_BUILD_CONFIG}")
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
            "Failed to build ${find_mode} installed-package consumer:\n${consumer_build_output}")
    endif()
    assert_no_sdk_path_leak("${consumer_build_output}" "${find_mode} verbose build output")
    file(GLOB_RECURSE consumer_link_metadata
        LIST_DIRECTORIES FALSE "${consumer_build_dir}/*link.txt")
    foreach(link_metadata IN LISTS consumer_link_metadata)
        file(READ "${link_metadata}" link_metadata_text)
        assert_no_sdk_path_leak("${link_metadata_text}" "${find_mode} link metadata")
    endforeach()

    set(consumer_test_command
        "${HOLOSCAN_CTEST_COMMAND}" --test-dir "${consumer_build_dir}" --output-on-failure
    )
    if(DEFINED HOLOSCAN_BUILD_CONFIG AND NOT HOLOSCAN_BUILD_CONFIG STREQUAL "")
        list(APPEND consumer_test_command -C "${HOLOSCAN_BUILD_CONFIG}")
    endif()
    execute_process(
        COMMAND ${consumer_test_command}
        RESULT_VARIABLE consumer_test_result
        OUTPUT_VARIABLE consumer_test_stdout
        ERROR_VARIABLE consumer_test_stderr
    )
    set(consumer_test_output "${consumer_test_stdout}\n${consumer_test_stderr}")
    assert_no_sdk_path_leak("${consumer_test_output}" "${find_mode} ctest output")
    if(NOT consumer_test_result EQUAL 0)
        message(FATAL_ERROR
            "${find_mode} installed-package consumer execution failed:\n"
            "${consumer_test_output}")
    endif()
endforeach()
