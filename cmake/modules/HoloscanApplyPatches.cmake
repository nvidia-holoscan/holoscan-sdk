# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

include(CMakeParseArguments)

# Construct an idempotent PATCH_COMMAND for ExternalProject, FetchContent, or CPM.
#
# Patches are interpreted as an ordered series. The generated command first
# validates the complete series against a temporary copy of the dependency
# source. It then applies the series only when every patch applies cleanly, or
# succeeds without changing the source when the complete series is already
# applied. A partially applied or conflicting series is rejected before the
# dependency source is modified.
#
# Usage:
#   find_package(holoscan REQUIRED) # Makes this function available to consumers.
#   holoscan_generate_dependency_patch_command(_patch_command
#       PATCHES
#           "${CMAKE_CURRENT_LIST_DIR}/patches/0001-first.patch"
#           "${CMAKE_CURRENT_LIST_DIR}/patches/0002-second.patch"
#   )
#   FetchContent_Declare(example
#       # ...
#       PATCH_COMMAND ${_patch_command}
#   )
#
# The output variable is a CMake list equivalent to:
#   /usr/bin/cmake \
#     -DHOLOSCAN_PATCH_GIT_EXECUTABLE:FILEPATH=/usr/bin/git \
#     -P /opt/holoscan/lib/cmake/holoscan/HoloscanApplyPatches.cmake \
#     -- \
#     /project/patches/0001-first.patch \
#     /project/patches/0002-second.patch
function(holoscan_generate_dependency_patch_command output_variable)
    set(options)
    set(one_value_args GIT_EXECUTABLE)
    set(multi_value_args PATCHES)
    cmake_parse_arguments(
        PARSE_ARGV 1 _HOLOSCAN_PATCH "${options}" "${one_value_args}" "${multi_value_args}")

    if(_HOLOSCAN_PATCH_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "holoscan_generate_dependency_patch_command received unknown arguments: "
            "${_HOLOSCAN_PATCH_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT output_variable)
        message(FATAL_ERROR
            "holoscan_generate_dependency_patch_command requires an output variable")
    endif()
    if(NOT _HOLOSCAN_PATCH_PATCHES)
        message(FATAL_ERROR
            "holoscan_generate_dependency_patch_command requires at least one patch in PATCHES")
    endif()

    if(NOT _HOLOSCAN_PATCH_GIT_EXECUTABLE)
        find_package(Git REQUIRED)
        set(_HOLOSCAN_PATCH_GIT_EXECUTABLE "${GIT_EXECUTABLE}")
    elseif(NOT EXISTS "${_HOLOSCAN_PATCH_GIT_EXECUTABLE}")
        message(FATAL_ERROR
            "holoscan_generate_dependency_patch_command GIT_EXECUTABLE does not exist: "
            "${_HOLOSCAN_PATCH_GIT_EXECUTABLE}")
    endif()

    set(_holoscan_patch_files)
    foreach(_holoscan_patch_file IN LISTS _HOLOSCAN_PATCH_PATCHES)
        get_filename_component(
            _holoscan_patch_file "${_holoscan_patch_file}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        if(NOT EXISTS "${_holoscan_patch_file}")
            message(FATAL_ERROR
                "holoscan_generate_dependency_patch_command patch does not exist: "
                "${_holoscan_patch_file}")
        endif()
        list(APPEND _holoscan_patch_files "${_holoscan_patch_file}")
    endforeach()

    set(_holoscan_patch_command
        "${CMAKE_COMMAND}"
        "-DHOLOSCAN_PATCH_GIT_EXECUTABLE:FILEPATH=${_HOLOSCAN_PATCH_GIT_EXECUTABLE}"
        -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
        --
        ${_holoscan_patch_files}
    )
    set("${output_variable}" "${_holoscan_patch_command}" PARENT_SCOPE)
endfunction()

function(_holoscan_run_git_apply result_variable error_variable source_directory)
    execute_process(
        COMMAND "${HOLOSCAN_PATCH_GIT_EXECUTABLE}" apply --ignore-whitespace ${ARGN}
        WORKING_DIRECTORY "${source_directory}"
        RESULT_VARIABLE _holoscan_git_apply_result
        OUTPUT_QUIET
        ERROR_VARIABLE _holoscan_git_apply_error
    )
    string(STRIP "${_holoscan_git_apply_error}" _holoscan_git_apply_error)
    set("${result_variable}" "${_holoscan_git_apply_result}" PARENT_SCOPE)
    set("${error_variable}" "${_holoscan_git_apply_error}" PARENT_SCOPE)
endfunction()

function(_holoscan_apply_patch_sequence result_variable error_variable source_directory)
    set(_holoscan_applied_patches)
    foreach(_holoscan_patch_file IN LISTS ARGN)
        _holoscan_run_git_apply(
            _holoscan_check_result _holoscan_check_error "${source_directory}"
            --check "${_holoscan_patch_file}")
        if(NOT _holoscan_check_result EQUAL 0)
            string(CONCAT _holoscan_apply_error
                "Patch does not apply cleanly: ${_holoscan_patch_file}\n"
                "${_holoscan_check_error}")
            break()
        endif()

        _holoscan_run_git_apply(
            _holoscan_apply_result _holoscan_apply_error "${source_directory}"
            "${_holoscan_patch_file}")
        if(NOT _holoscan_apply_result EQUAL 0)
            set(_holoscan_apply_error
                "Failed to apply patch: ${_holoscan_patch_file}\n${_holoscan_apply_error}")
            break()
        endif()
        list(APPEND _holoscan_applied_patches "${_holoscan_patch_file}")
    endforeach()

    if(DEFINED _holoscan_apply_error AND NOT _holoscan_apply_error STREQUAL "")
        list(REVERSE _holoscan_applied_patches)
        foreach(_holoscan_patch_file IN LISTS _holoscan_applied_patches)
            _holoscan_run_git_apply(
                _holoscan_rollback_result _holoscan_rollback_error "${source_directory}"
                --reverse "${_holoscan_patch_file}")
            if(NOT _holoscan_rollback_result EQUAL 0)
                string(APPEND _holoscan_apply_error
                    "\nFailed to restore the validation copy after ${_holoscan_patch_file}:\n"
                    "${_holoscan_rollback_error}")
                break()
            endif()
        endforeach()
        set("${result_variable}" 1 PARENT_SCOPE)
        set("${error_variable}" "${_holoscan_apply_error}" PARENT_SCOPE)
        return()
    endif()

    set("${result_variable}" 0 PARENT_SCOPE)
    set("${error_variable}" "" PARENT_SCOPE)
endfunction()

function(_holoscan_reverse_patch_sequence result_variable error_variable source_directory)
    set(_holoscan_patch_files ${ARGN})
    list(REVERSE _holoscan_patch_files)
    foreach(_holoscan_patch_file IN LISTS _holoscan_patch_files)
        _holoscan_run_git_apply(
            _holoscan_check_result _holoscan_check_error "${source_directory}"
            --reverse --check "${_holoscan_patch_file}")
        if(NOT _holoscan_check_result EQUAL 0)
            string(CONCAT _holoscan_reverse_error
                "Patch is not cleanly reversible: ${_holoscan_patch_file}\n"
                "${_holoscan_check_error}")
            set("${result_variable}" 1 PARENT_SCOPE)
            set("${error_variable}" "${_holoscan_reverse_error}" PARENT_SCOPE)
            return()
        endif()

        _holoscan_run_git_apply(
            _holoscan_reverse_result _holoscan_reverse_error "${source_directory}"
            --reverse "${_holoscan_patch_file}")
        if(NOT _holoscan_reverse_result EQUAL 0)
            set("${result_variable}" 1 PARENT_SCOPE)
            set("${error_variable}"
                "Failed to reverse patch: ${_holoscan_patch_file}\n${_holoscan_reverse_error}"
                PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set("${result_variable}" 0 PARENT_SCOPE)
    set("${error_variable}" "" PARENT_SCOPE)
endfunction()

# A single patch can be checked directly because git apply is all-or-nothing.
# For an ordered series, work against a temporary source copy first:
#   1. Apply every patch forward to identify a clean, unpatched source.
#   2. If forward validation fails, reverse every patch from last to first to
#      identify a fully patched source. The forward helper restores the copy
#      before this reverse check when only part of the series applied.
#   3. Apply to the real source only after full forward validation; otherwise
#      accept the fully patched state or reject the inconsistent state.
function(_holoscan_apply_patches source_directory)
    set(_holoscan_patch_files ${ARGN})

    if(NOT IS_DIRECTORY "${source_directory}")
        message(FATAL_ERROR "Dependency patch source directory does not exist: ${source_directory}")
    endif()

    list(LENGTH _holoscan_patch_files _holoscan_patch_count)
    if(_holoscan_patch_count EQUAL 1)
        list(GET _holoscan_patch_files 0 _holoscan_patch_file)
        _holoscan_run_git_apply(
            _holoscan_forward_result _holoscan_forward_error "${source_directory}"
            --check "${_holoscan_patch_file}")
        if(_holoscan_forward_result EQUAL 0)
            _holoscan_run_git_apply(
                _holoscan_apply_result _holoscan_apply_error "${source_directory}"
                "${_holoscan_patch_file}")
            if(NOT _holoscan_apply_result EQUAL 0)
                message(FATAL_ERROR
                    "Dependency patch application failed after successful validation:\n"
                    "${_holoscan_apply_error}")
            endif()
            return()
        endif()

        _holoscan_run_git_apply(
            _holoscan_reverse_result _holoscan_reverse_error "${source_directory}"
            --reverse --check "${_holoscan_patch_file}")
        if(_holoscan_reverse_result EQUAL 0)
            return()
        endif()
        message(FATAL_ERROR
            "Dependency patch is neither applicable nor already applied: "
            "${_holoscan_patch_file}\n"
            "Forward validation:\n${_holoscan_forward_error}\n"
            "Reverse validation:\n${_holoscan_reverse_error}")
    endif()

    string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef _holoscan_patch_suffix)
    get_filename_component(_holoscan_patch_parent "${source_directory}" DIRECTORY)
    set(_holoscan_validation_directory
        "${_holoscan_patch_parent}/.holoscan-patch-${_holoscan_patch_suffix}")
    file(MAKE_DIRECTORY "${_holoscan_validation_directory}")
    file(COPY "${source_directory}/"
        DESTINATION "${_holoscan_validation_directory}"
        PATTERN ".git" EXCLUDE)

    _holoscan_apply_patch_sequence(
        _holoscan_forward_result _holoscan_forward_error
        "${_holoscan_validation_directory}" ${_holoscan_patch_files})
    if(_holoscan_forward_result EQUAL 0)
        file(REMOVE_RECURSE "${_holoscan_validation_directory}")
        _holoscan_apply_patch_sequence(
            _holoscan_apply_result _holoscan_apply_error
            "${source_directory}" ${_holoscan_patch_files})
        if(NOT _holoscan_apply_result EQUAL 0)
            message(FATAL_ERROR
                "Dependency patch application failed after successful validation:\n"
                "${_holoscan_apply_error}")
        endif()
        return()
    endif()

    _holoscan_reverse_patch_sequence(
        _holoscan_reverse_result _holoscan_reverse_error
        "${_holoscan_validation_directory}" ${_holoscan_patch_files})
    file(REMOVE_RECURSE "${_holoscan_validation_directory}")
    if(_holoscan_reverse_result EQUAL 0)
        return()
    endif()

    message(FATAL_ERROR
        "Dependency patch series is partially applied, conflicts with the source, or is ordered "
        "incorrectly.\n"
        "Forward validation:\n${_holoscan_forward_error}\n"
        "Reverse validation:\n${_holoscan_reverse_error}")
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    if(NOT HOLOSCAN_PATCH_GIT_EXECUTABLE OR
       NOT EXISTS "${HOLOSCAN_PATCH_GIT_EXECUTABLE}")
        message(FATAL_ERROR
            "HoloscanApplyPatches.cmake requires HOLOSCAN_PATCH_GIT_EXECUTABLE")
    endif()

    set(_holoscan_patch_files)
    set(_holoscan_collect_patch_files FALSE)
    math(EXPR _holoscan_last_argument "${CMAKE_ARGC} - 1")
    foreach(_holoscan_argument_index RANGE 0 ${_holoscan_last_argument})
        set(_holoscan_argument "${CMAKE_ARGV${_holoscan_argument_index}}")
        if(_holoscan_collect_patch_files)
            list(APPEND _holoscan_patch_files "${_holoscan_argument}")
        elseif(_holoscan_argument STREQUAL "--")
            set(_holoscan_collect_patch_files TRUE)
        endif()
    endforeach()
    if(NOT _holoscan_patch_files)
        message(FATAL_ERROR "HoloscanApplyPatches.cmake requires at least one patch file")
    endif()
    foreach(_holoscan_patch_file IN LISTS _holoscan_patch_files)
        if(NOT EXISTS "${_holoscan_patch_file}")
            message(FATAL_ERROR "Dependency patch does not exist: ${_holoscan_patch_file}")
        endif()
    endforeach()

    _holoscan_apply_patches("${CMAKE_CURRENT_BINARY_DIR}" ${_holoscan_patch_files})
endif()
