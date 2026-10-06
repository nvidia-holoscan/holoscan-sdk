# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_ARTIFACT_FIXTURE_SOURCE_DIR
        HOLOSCAN_ARTIFACT_FIXTURE_WORK_DIR
        HOLOSCAN_GRAPH_ARTIFACTS_MODULE
        HOLOSCAN_ARTIFACT_FIXTURE_MODE
        HOLOSCAN_GENERATOR)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

set(fixture_build_dir "${HOLOSCAN_ARTIFACT_FIXTURE_WORK_DIR}/build")
set(fixture_install_prefix "${HOLOSCAN_ARTIFACT_FIXTURE_WORK_DIR}/install")
set(fixture_absolute_destination "${HOLOSCAN_ARTIFACT_FIXTURE_WORK_DIR}/absolute-destination")
file(REMOVE_RECURSE "${HOLOSCAN_ARTIFACT_FIXTURE_WORK_DIR}")

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${HOLOSCAN_ARTIFACT_FIXTURE_SOURCE_DIR}"
    -B "${fixture_build_dir}"
    -G "${HOLOSCAN_GENERATOR}"
    "-DHOLOSCAN_GRAPH_ARTIFACTS_MODULE:FILEPATH=${HOLOSCAN_GRAPH_ARTIFACTS_MODULE}"
    "-DHOLOSCAN_ARTIFACT_FIXTURE_MODE:STRING=${HOLOSCAN_ARTIFACT_FIXTURE_MODE}"
    "-DHOLOSCAN_ABSOLUTE_INSTALL_DESTINATION:PATH=${fixture_absolute_destination}"
)
if(HOLOSCAN_ARTIFACT_FIXTURE_MODE STREQUAL "cross_without_emulator")
    list(APPEND configure_command "-DCMAKE_SYSTEM_NAME:STRING=Generic")
endif()
if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
endif()
if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
endif()
if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
    list(APPEND configure_command "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
endif()
set(toolchain_selects_compiler FALSE)
if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    set(toolchain_selects_compiler TRUE)
endif()
if(NOT HOLOSCAN_GENERATOR MATCHES "^Visual Studio" AND NOT toolchain_selects_compiler AND
   DEFINED HOLOSCAN_CXX_COMPILER AND NOT HOLOSCAN_CXX_COMPILER STREQUAL "")
    list(APPEND configure_command "-DCMAKE_CXX_COMPILER:FILEPATH=${HOLOSCAN_CXX_COMPILER}")
endif()
if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    list(APPEND configure_command "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${HOLOSCAN_TOOLCHAIN_FILE}")
endif()
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND configure_command "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
endif()
if(DEFINED HOLOSCAN_MSVC_RUNTIME_LIBRARY AND NOT HOLOSCAN_MSVC_RUNTIME_LIBRARY STREQUAL "")
    list(APPEND configure_command
        "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${HOLOSCAN_MSVC_RUNTIME_LIBRARY}")
endif()

execute_process(
    COMMAND ${configure_command}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_stdout
    ERROR_VARIABLE configure_stderr
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to configure invalid artifact fixture:\n${configure_stdout}\n${configure_stderr}")
endif()

if(NOT HOLOSCAN_ARTIFACT_FIXTURE_MODE STREQUAL "cross_without_emulator")
    set(build_command "${CMAKE_COMMAND}" --build "${fixture_build_dir}")
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND build_command --config "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(
        COMMAND ${build_command}
        RESULT_VARIABLE build_result
        OUTPUT_VARIABLE build_stdout
        ERROR_VARIABLE build_stderr
    )
    if(NOT build_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to build artifact fixture:\n${build_stdout}\n${build_stderr}")
    endif()
endif()

if(HOLOSCAN_ARTIFACT_FIXTURE_MODE STREQUAL "valid_absolute_destination" OR
   HOLOSCAN_ARTIFACT_FIXTURE_MODE STREQUAL "cross_without_emulator")
    set(artifact_target_command
        "${CMAKE_COMMAND}" --build "${fixture_build_dir}"
        --target artifact_install_fixture_graph_artifacts
    )
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND artifact_target_command --config "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(
        COMMAND ${artifact_target_command}
        RESULT_VARIABLE artifact_target_result
        OUTPUT_VARIABLE artifact_target_stdout
        ERROR_VARIABLE artifact_target_stderr
    )
    set(artifact_target_output "${artifact_target_stdout}\n${artifact_target_stderr}")
    if(HOLOSCAN_ARTIFACT_FIXTURE_MODE STREQUAL "valid_absolute_destination")
        if(NOT artifact_target_result EQUAL 0)
            message(FATAL_ERROR
                "Explicit artifact target failed:\n${artifact_target_output}")
        endif()
    else()
        if(artifact_target_result EQUAL 0)
            message(FATAL_ERROR
                "Explicit artifact target accepted a cross-compiled executable without an emulator")
        endif()
        string(FIND "${artifact_target_output}" "without CROSSCOMPILING_EMULATOR"
            artifact_target_diagnostic_position)
        if(artifact_target_diagnostic_position EQUAL -1)
            message(FATAL_ERROR
                "Explicit artifact target failed without the expected diagnostic:\n"
                "${artifact_target_output}")
        endif()
    endif()
endif()

if(HOLOSCAN_ARTIFACT_FIXTURE_MODE STREQUAL "default_install_excluded")
    set(default_install_command
        "${CMAKE_COMMAND}" --install "${fixture_build_dir}"
        --prefix "${fixture_install_prefix}"
    )
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND default_install_command --config "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(
        COMMAND ${default_install_command}
        RESULT_VARIABLE default_install_result
        OUTPUT_VARIABLE default_install_stdout
        ERROR_VARIABLE default_install_stderr
    )
    if(NOT default_install_result EQUAL 0)
        message(FATAL_ERROR
            "Default install unexpectedly executed artifact generation:\n"
            "${default_install_stdout}\n${default_install_stderr}")
    endif()
    if(EXISTS "${fixture_install_prefix}/share/holoscan/graph-artifact-install-fixture")
        message(FATAL_ERROR "Default install included the optional graph-artifact component")
    endif()
    return()
endif()

set(install_command
    "${CMAKE_COMMAND}" --install "${fixture_build_dir}"
    --prefix "${fixture_install_prefix}"
    --component holoscan-graph-artifacts
)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND install_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(
    COMMAND ${install_command}
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_stdout
    ERROR_VARIABLE install_stderr
)
if(HOLOSCAN_ARTIFACT_FIXTURE_MODE STREQUAL "valid_absolute_destination")
    if(NOT install_result EQUAL 0)
        message(FATAL_ERROR
            "Absolute artifact installation failed:\n${install_stdout}\n${install_stderr}")
    endif()
    set(expected_bundle "${fixture_absolute_destination}/fixture-graph-id/developer")
    foreach(expected_file IN ITEMS
            artifact-manifest.json
            endpoint-catalog.json
            execution-plan.json
            profile-manifest.json)
        if(NOT EXISTS "${expected_bundle}/${expected_file}" OR
           IS_DIRECTORY "${expected_bundle}/${expected_file}" OR
           IS_SYMLINK "${expected_bundle}/${expected_file}")
            message(FATAL_ERROR
                "Absolute artifact installation omitted ${expected_bundle}/${expected_file}")
        endif()
    endforeach()
    if(EXISTS "${fixture_install_prefix}/${fixture_absolute_destination}")
        message(FATAL_ERROR "Absolute artifact destination was incorrectly prefixed")
    endif()
    return()
endif()

if(install_result EQUAL 0)
    message(FATAL_ERROR
        "Artifact installation accepted invalid ${HOLOSCAN_ARTIFACT_FIXTURE_MODE} case")
endif()

set(install_output "${install_stdout}\n${install_stderr}")
set(expected_diagnostic "Artifact generation for artifact_install_fixture")
if(HOLOSCAN_ARTIFACT_FIXTURE_MODE STREQUAL "cross_without_emulator")
    set(expected_diagnostic "without CROSSCOMPILING_EMULATOR")
endif()
string(FIND "${install_output}" "${expected_diagnostic}" artifact_diagnostic_position)
if(artifact_diagnostic_position EQUAL -1)
    message(FATAL_ERROR
        "Artifact installation failed without the expected diagnostic:\n"
        "${install_output}")
endif()
