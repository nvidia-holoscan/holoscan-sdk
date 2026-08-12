# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

include(CMakeParseArguments)

# Add an explicit build target that asks an application to publish its frozen Graph artifacts.
#
# The executable remains the single owner of Graph composition and CompileOptions. It must accept
# the following cold command-line contract:
#
#   --emit-artifacts --artifact-root <path> --deployment-id <id>
#   [--deployment-config <path>]
#
# The command is intentionally not part of ALL: artifact publication is an explicit deployment
# operation and must never start application lifecycle or transport services as a build side effect.
# When INSTALL_DESTINATION is present, explicit component installation regenerates into a private
# clean staging root, accepts exactly one graph/deployment directory containing the four canonical
# files, and installs only those files. The optional component is excluded from an unqualified
# install so ordinary SDK packaging never executes an application as a side effect. Cross-compiled
# executables require CROSSCOMPILING_EMULATOR when either the explicit target or installation
# component is invoked.
function(holoscan_add_graph_artifacts)
    set(options)
    set(one_value_args
        TARGET
        DEPLOYMENT_CONFIG
        DEPLOYMENT_ID
        OUTPUT_DIRECTORY
        INSTALL_DESTINATION
    )
    cmake_parse_arguments(HAGA "${options}" "${one_value_args}" "" ${ARGN})

    if(HAGA_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "holoscan_add_graph_artifacts received unknown arguments: ${HAGA_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT HAGA_TARGET)
        message(FATAL_ERROR "holoscan_add_graph_artifacts requires TARGET")
    endif()
    if(NOT TARGET "${HAGA_TARGET}")
        message(FATAL_ERROR
            "holoscan_add_graph_artifacts TARGET '${HAGA_TARGET}' does not exist")
    endif()
    get_target_property(_holoscan_artifact_target_type "${HAGA_TARGET}" TYPE)
    if(NOT _holoscan_artifact_target_type STREQUAL "EXECUTABLE")
        message(FATAL_ERROR
            "holoscan_add_graph_artifacts TARGET '${HAGA_TARGET}' must be an executable")
    endif()
    get_target_property(_holoscan_artifact_target_imported "${HAGA_TARGET}" IMPORTED)
    if(_holoscan_artifact_target_imported)
        message(FATAL_ERROR
            "holoscan_add_graph_artifacts TARGET '${HAGA_TARGET}' must not be imported")
    endif()

    if(NOT HAGA_DEPLOYMENT_ID)
        set(HAGA_DEPLOYMENT_ID "default")
    endif()
    if(NOT HAGA_OUTPUT_DIRECTORY)
        if(DEFINED ENV{HOLOSCAN_ARTIFACT_DIR} AND NOT "$ENV{HOLOSCAN_ARTIFACT_DIR}" STREQUAL "")
            set(HAGA_OUTPUT_DIRECTORY "$ENV{HOLOSCAN_ARTIFACT_DIR}")
        else()
            set(HAGA_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/holoscan-artifacts")
        endif()
    endif()
    get_filename_component(
        _holoscan_artifact_output "${HAGA_OUTPUT_DIRECTORY}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")

    set(_holoscan_artifact_runner "$<TARGET_FILE:${HAGA_TARGET}>")
    set(_holoscan_artifact_cross_without_emulator FALSE)
    if(CMAKE_CROSSCOMPILING)
        get_target_property(
            _holoscan_artifact_cross_emulator
            "${HAGA_TARGET}"
            CROSSCOMPILING_EMULATOR
        )
        if(NOT _holoscan_artifact_cross_emulator OR
           _holoscan_artifact_cross_emulator MATCHES "-NOTFOUND$")
            set(_holoscan_artifact_cross_without_emulator TRUE)
        else()
            set(_holoscan_artifact_runner
                ${_holoscan_artifact_cross_emulator}
                "$<TARGET_FILE:${HAGA_TARGET}>"
            )
        endif()
    endif()

    set(_holoscan_artifact_command
        ${_holoscan_artifact_runner}
        --emit-artifacts
        --artifact-root "${_holoscan_artifact_output}"
        --deployment-id "${HAGA_DEPLOYMENT_ID}"
    )
    set(_holoscan_artifact_dependencies "${HAGA_TARGET}")
    if(HAGA_DEPLOYMENT_CONFIG)
        get_filename_component(
            _holoscan_deployment_config "${HAGA_DEPLOYMENT_CONFIG}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        if(NOT EXISTS "${_holoscan_deployment_config}")
            message(FATAL_ERROR
                "holoscan_add_graph_artifacts DEPLOYMENT_CONFIG does not exist: "
                "${_holoscan_deployment_config}")
        endif()
        list(APPEND _holoscan_artifact_command
            --deployment-config "${_holoscan_deployment_config}")
        list(APPEND _holoscan_artifact_dependencies "${_holoscan_deployment_config}")
    endif()

    set(_holoscan_artifact_target "${HAGA_TARGET}_graph_artifacts")
    if(TARGET "${_holoscan_artifact_target}")
        message(FATAL_ERROR
            "holoscan_add_graph_artifacts already registered ${_holoscan_artifact_target}")
    endif()
    if(_holoscan_artifact_cross_without_emulator)
        add_custom_target("${_holoscan_artifact_target}"
            COMMAND "${CMAKE_COMMAND}" -E echo
                "Cannot execute cross-compiled TARGET '${HAGA_TARGET}' without CROSSCOMPILING_EMULATOR"
            COMMAND "${CMAKE_COMMAND}" -E false
            DEPENDS ${_holoscan_artifact_dependencies}
            VERBATIM
            COMMENT
                "Rejecting ${HAGA_TARGET} Graph artifact generation without an emulator"
        )
    else()
        add_custom_target("${_holoscan_artifact_target}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${_holoscan_artifact_output}"
            COMMAND ${_holoscan_artifact_command}
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:${HAGA_TARGET}>"
            DEPENDS ${_holoscan_artifact_dependencies}
            VERBATIM
            COMMENT
                "Publishing ${HAGA_TARGET} Graph artifacts for deployment ${HAGA_DEPLOYMENT_ID}"
        )
    endif()
    set_target_properties("${_holoscan_artifact_target}" PROPERTIES
        HOLOSCAN_ARTIFACT_DIRECTORY "${_holoscan_artifact_output}"
        HOLOSCAN_ARTIFACT_DEPLOYMENT_ID "${HAGA_DEPLOYMENT_ID}"
    )

    if(HAGA_INSTALL_DESTINATION)
        # CMake installation does not build custom targets. Generate into a private clean root at
        # install time so a component install cannot silently package no bundle or stale graph IDs
        # left by a previous explicit artifact-target invocation.
        set(_holoscan_artifact_install_output
            "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_holoscan_artifact_target}-install/$<CONFIG>"
        )
        set(_holoscan_artifact_install_deployment_argument "")
        if(HAGA_DEPLOYMENT_CONFIG)
            set(_holoscan_artifact_install_deployment_argument
                "        --deployment-config \"${_holoscan_deployment_config}\""
            )
        endif()
        set(_holoscan_artifact_install_runner_entries "")
        foreach(_holoscan_artifact_runner_argument IN LISTS _holoscan_artifact_runner)
            string(APPEND _holoscan_artifact_install_runner_entries
                "    [==[${_holoscan_artifact_runner_argument}]==]\n")
        endforeach()
        set(_holoscan_artifact_install_cross_guard "")
        if(_holoscan_artifact_cross_without_emulator)
            set(_holoscan_artifact_install_cross_guard
                "message(FATAL_ERROR \"Cannot install Graph artifacts for cross-compiled TARGET "
                "'${HAGA_TARGET}' without CROSSCOMPILING_EMULATOR\")"
            )
        endif()
        if(IS_ABSOLUTE "${HAGA_INSTALL_DESTINATION}")
            set(_holoscan_artifact_install_destination
                "[==[${HAGA_INSTALL_DESTINATION}]==]")
        else()
            set(_holoscan_artifact_install_destination
                "\"\${CMAKE_INSTALL_PREFIX}/${HAGA_INSTALL_DESTINATION}\"")
        endif()
        set(_holoscan_artifact_install_code [=[
@_holoscan_artifact_install_cross_guard@
set(_holoscan_artifact_root "@_holoscan_artifact_install_output@")
file(REMOVE_RECURSE "${_holoscan_artifact_root}")
file(MAKE_DIRECTORY "${_holoscan_artifact_root}")
set(_holoscan_artifact_runner
@_holoscan_artifact_install_runner_entries@)
execute_process(
    COMMAND ${_holoscan_artifact_runner}
        --emit-artifacts
        --artifact-root "${_holoscan_artifact_root}"
        --deployment-id "@HAGA_DEPLOYMENT_ID@"
@_holoscan_artifact_install_deployment_argument@
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:@HAGA_TARGET@>"
    RESULT_VARIABLE _holoscan_artifact_result
    OUTPUT_VARIABLE _holoscan_artifact_stdout
    ERROR_VARIABLE _holoscan_artifact_stderr
)
if(NOT _holoscan_artifact_result EQUAL 0)
    message(FATAL_ERROR
        "Artifact generation failed for @HAGA_TARGET@: "
        "${_holoscan_artifact_stderr}${_holoscan_artifact_stdout}")
endif()

file(GLOB _holoscan_graph_entries LIST_DIRECTORIES TRUE "${_holoscan_artifact_root}/*")
list(LENGTH _holoscan_graph_entries _holoscan_graph_entry_count)
if(NOT _holoscan_graph_entry_count EQUAL 1)
    message(FATAL_ERROR
        "Artifact generation for @HAGA_TARGET@ produced "
        "${_holoscan_graph_entry_count} graph-id entries instead of one")
endif()
list(GET _holoscan_graph_entries 0 _holoscan_graph_directory)
if(NOT IS_DIRECTORY "${_holoscan_graph_directory}" OR
   IS_SYMLINK "${_holoscan_graph_directory}")
    message(FATAL_ERROR
        "Artifact generation for @HAGA_TARGET@ produced a graph-id entry that is not a regular "
        "directory: ${_holoscan_graph_directory}")
endif()

set(_holoscan_bundle_directory "${_holoscan_graph_directory}/@HAGA_DEPLOYMENT_ID@")
file(GLOB _holoscan_deployment_entries
    LIST_DIRECTORIES TRUE
    RELATIVE "${_holoscan_graph_directory}"
    "${_holoscan_graph_directory}/*"
)
if(NOT "${_holoscan_deployment_entries}" STREQUAL "@HAGA_DEPLOYMENT_ID@")
    message(FATAL_ERROR
        "Artifact generation for @HAGA_TARGET@ produced an invalid deployment hierarchy: "
        "${_holoscan_deployment_entries}")
endif()
if(NOT IS_DIRECTORY "${_holoscan_bundle_directory}" OR
   IS_SYMLINK "${_holoscan_bundle_directory}")
    message(FATAL_ERROR
        "Artifact generation for @HAGA_TARGET@ produced a deployment entry that is not a "
        "regular directory: ${_holoscan_bundle_directory}")
endif()
set(_holoscan_expected_files
    artifact-manifest.json
    endpoint-catalog.json
    execution-plan.json
    profile-manifest.json
)
file(GLOB _holoscan_bundle_entries
    LIST_DIRECTORIES TRUE
    RELATIVE "${_holoscan_bundle_directory}"
    "${_holoscan_bundle_directory}/*"
)
list(SORT _holoscan_bundle_entries)
list(SORT _holoscan_expected_files)
if(NOT "${_holoscan_bundle_entries}" STREQUAL "${_holoscan_expected_files}")
    message(FATAL_ERROR
        "Artifact generation for @HAGA_TARGET@ produced an invalid bundle: "
        "${_holoscan_bundle_entries}")
endif()

set(_holoscan_bundle_paths)
foreach(_holoscan_expected_file IN LISTS _holoscan_expected_files)
    set(_holoscan_bundle_path "${_holoscan_bundle_directory}/${_holoscan_expected_file}")
    if(NOT EXISTS "${_holoscan_bundle_path}" OR
       IS_DIRECTORY "${_holoscan_bundle_path}" OR
       IS_SYMLINK "${_holoscan_bundle_path}")
        message(FATAL_ERROR
            "Artifact generation for @HAGA_TARGET@ produced a canonical entry that is not a "
            "regular file: ${_holoscan_bundle_path}")
    endif()
    # CMake has no portable IS_REGULAR_FILE predicate. Generated canonical JSON is always
    # nonempty, while filesystem special nodes such as FIFOs report a zero stat size. Requiring a
    # positive size closes that special-node gap without opening or potentially blocking on it.
    file(SIZE "${_holoscan_bundle_path}" _holoscan_bundle_size)
    if(_holoscan_bundle_size EQUAL 0)
        message(FATAL_ERROR
            "Artifact generation for @HAGA_TARGET@ produced a canonical entry that is not a "
            "nonempty regular file: ${_holoscan_bundle_path}")
    endif()
    list(APPEND _holoscan_bundle_paths "${_holoscan_bundle_path}")
endforeach()
get_filename_component(_holoscan_graph_id "${_holoscan_graph_directory}" NAME)
set(_holoscan_install_destination @_holoscan_artifact_install_destination@)
file(INSTALL
    DESTINATION
        "${_holoscan_install_destination}/${_holoscan_graph_id}/@HAGA_DEPLOYMENT_ID@"
    TYPE FILE
    FILES ${_holoscan_bundle_paths}
)
]=])
        string(CONFIGURE
            "${_holoscan_artifact_install_code}"
            _holoscan_artifact_install_code
            @ONLY
        )
        install(CODE "${_holoscan_artifact_install_code}"
            COMPONENT holoscan-graph-artifacts
            EXCLUDE_FROM_ALL
        )
    endif()

    unset(_holoscan_artifact_command)
    unset(_holoscan_artifact_dependencies)
    unset(_holoscan_artifact_install_code)
    unset(_holoscan_artifact_install_destination)
    unset(_holoscan_artifact_install_deployment_argument)
    unset(_holoscan_artifact_install_output)
    unset(_holoscan_artifact_install_runner_entries)
    unset(_holoscan_artifact_output)
    unset(_holoscan_artifact_target)
    unset(_holoscan_artifact_target_imported)
    unset(_holoscan_artifact_target_type)
    unset(_holoscan_deployment_config)
    unset(_holoscan_artifact_cross_emulator)
    unset(_holoscan_artifact_emulator_argument)
endfunction()
