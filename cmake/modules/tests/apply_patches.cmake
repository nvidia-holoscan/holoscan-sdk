# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

# Keep the module under test explicit so this script can validate either the
# source-tree module or an installed copy without relying on CMAKE_MODULE_PATH.
if(NOT HOLOSCAN_APPLY_PATCHES_MODULE OR
   NOT EXISTS "${HOLOSCAN_APPLY_PATCHES_MODULE}")
    message(FATAL_ERROR "HOLOSCAN_APPLY_PATCHES_MODULE must identify the module under test")
endif()
if(NOT HOLOSCAN_PATCH_TEST_ROOT)
    message(FATAL_ERROR "HOLOSCAN_PATCH_TEST_ROOT is required")
endif()

include("${HOLOSCAN_APPLY_PATCHES_MODULE}")

function(_holoscan_assert_content path expected_content)
    file(READ "${path}" _holoscan_actual_content)
    if(NOT _holoscan_actual_content STREQUAL expected_content)
        message(FATAL_ERROR
            "Unexpected content in ${path}:\n"
            "expected=[==[${expected_content}]==]\n"
            "actual=[==[${_holoscan_actual_content}]==]")
    endif()
endfunction()

function(_holoscan_run_patch_command source_directory expected_result)
    holoscan_generate_dependency_patch_command(_holoscan_patch_command PATCHES ${ARGN})
    execute_process(
        COMMAND ${_holoscan_patch_command}
        WORKING_DIRECTORY "${source_directory}"
        RESULT_VARIABLE _holoscan_patch_result
        OUTPUT_VARIABLE _holoscan_patch_output
        ERROR_VARIABLE _holoscan_patch_error
    )
    if(expected_result STREQUAL "SUCCESS" AND NOT _holoscan_patch_result EQUAL 0)
        message(FATAL_ERROR
            "Patch command unexpectedly failed:\n"
            "${_holoscan_patch_output}\n${_holoscan_patch_error}")
    elseif(expected_result STREQUAL "FAILURE" AND _holoscan_patch_result EQUAL 0)
        message(FATAL_ERROR "Patch command unexpectedly succeeded")
    endif()

    get_filename_component(_holoscan_source_parent "${source_directory}" DIRECTORY)
    file(GLOB _holoscan_validation_directories
        "${_holoscan_source_parent}/.holoscan-patch-*")
    if(_holoscan_validation_directories)
        message(FATAL_ERROR
            "Patch command left validation directories behind: "
            "${_holoscan_validation_directories}")
    endif()
endfunction()

file(REMOVE_RECURSE "${HOLOSCAN_PATCH_TEST_ROOT}")
file(MAKE_DIRECTORY "${HOLOSCAN_PATCH_TEST_ROOT}")

# A one-patch command applies to an archive-like directory without Git metadata
# and is idempotent when invoked again.
set(_holoscan_single_root "${HOLOSCAN_PATCH_TEST_ROOT}/single")
set(_holoscan_single_source "${_holoscan_single_root}/source")
file(MAKE_DIRECTORY "${_holoscan_single_source}")
file(WRITE "${_holoscan_single_source}/value.txt" "before\n")
file(WRITE "${_holoscan_single_root}/change.patch" [=[
diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-before
+after
]=])
_holoscan_run_patch_command(
    "${_holoscan_single_source}" SUCCESS "${_holoscan_single_root}/change.patch")
_holoscan_assert_content("${_holoscan_single_source}/value.txt" "after\n")
_holoscan_run_patch_command(
    "${_holoscan_single_source}" SUCCESS "${_holoscan_single_root}/change.patch")
_holoscan_assert_content("${_holoscan_single_source}/value.txt" "after\n")

# Later patches may depend on context introduced by earlier patches. Validate
# and apply in forward order, then detect an already-applied series by reversing
# it in the validation copy.
set(_holoscan_series_root "${HOLOSCAN_PATCH_TEST_ROOT}/series")
set(_holoscan_series_source "${_holoscan_series_root}/source")
file(MAKE_DIRECTORY "${_holoscan_series_source}")
file(WRITE "${_holoscan_series_source}/value.txt" "alpha\n")
file(WRITE "${_holoscan_series_root}/0001-alpha-to-beta.patch" [=[
diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-alpha
+beta
]=])
file(WRITE "${_holoscan_series_root}/0002-beta-to-gamma.patch" [=[
diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-beta
+gamma
]=])
set(_holoscan_series_patches
    "${_holoscan_series_root}/0001-alpha-to-beta.patch"
    "${_holoscan_series_root}/0002-beta-to-gamma.patch"
)
_holoscan_run_patch_command(
    "${_holoscan_series_source}" SUCCESS ${_holoscan_series_patches})
_holoscan_assert_content("${_holoscan_series_source}/value.txt" "gamma\n")
_holoscan_run_patch_command(
    "${_holoscan_series_source}" SUCCESS ${_holoscan_series_patches})
_holoscan_assert_content("${_holoscan_series_source}/value.txt" "gamma\n")

# Reject a partial series without changing the dependency source.
set(_holoscan_partial_root "${HOLOSCAN_PATCH_TEST_ROOT}/partial")
set(_holoscan_partial_source "${_holoscan_partial_root}/source")
file(MAKE_DIRECTORY "${_holoscan_partial_source}")
file(WRITE "${_holoscan_partial_source}/value.txt" "beta\n")
_holoscan_run_patch_command(
    "${_holoscan_partial_source}" FAILURE ${_holoscan_series_patches})
_holoscan_assert_content("${_holoscan_partial_source}/value.txt" "beta\n")

# Exercise the public command through FetchContent's URL/archive path. The
# dependency has no Git repository, and its CMakeLists verifies that patching
# completed before FetchContent adds the source directory.
set(_holoscan_archive_root "${HOLOSCAN_PATCH_TEST_ROOT}/archive")
set(_holoscan_archive_input "${_holoscan_archive_root}/input/archive-dependency")
file(MAKE_DIRECTORY "${_holoscan_archive_input}")
file(WRITE "${_holoscan_archive_input}/value.txt" "archive-before\n")
file(WRITE "${_holoscan_archive_input}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 4.0)
project(archive_dependency NONE)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/value.txt" _archive_value)
if(NOT _archive_value STREQUAL "archive-after\n")
    message(FATAL_ERROR "Archive dependency was not patched: ${_archive_value}")
endif()
]=])
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E tar cf
        "${_holoscan_archive_root}/archive-dependency.tar"
        --format=gnutar archive-dependency
    WORKING_DIRECTORY "${_holoscan_archive_root}/input"
    COMMAND_ERROR_IS_FATAL ANY
)
file(WRITE "${_holoscan_archive_root}/archive.patch" [=[
diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-archive-before
+archive-after
]=])
file(WRITE "${_holoscan_archive_root}/consumer/CMakeLists.txt.in" [=[
cmake_minimum_required(VERSION 4.0)
project(archive_consumer NONE)
include(FetchContent)
include("@HOLOSCAN_APPLY_PATCHES_MODULE@")
holoscan_generate_dependency_patch_command(_archive_patch_command
    PATCHES "@HOLOSCAN_ARCHIVE_PATCH@")
FetchContent_Declare(archive_dependency
    URL "@HOLOSCAN_ARCHIVE_URL@"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    PATCH_COMMAND ${_archive_patch_command}
)
FetchContent_MakeAvailable(archive_dependency)
]=])
set(HOLOSCAN_ARCHIVE_PATCH "${_holoscan_archive_root}/archive.patch")
set(HOLOSCAN_ARCHIVE_URL "file://${_holoscan_archive_root}/archive-dependency.tar")
configure_file(
    "${_holoscan_archive_root}/consumer/CMakeLists.txt.in"
    "${_holoscan_archive_root}/consumer/CMakeLists.txt"
    @ONLY
)
execute_process(
    COMMAND "${CMAKE_COMMAND}"
        -S "${_holoscan_archive_root}/consumer"
        -B "${_holoscan_archive_root}/consumer-build"
    RESULT_VARIABLE _holoscan_archive_result
    OUTPUT_VARIABLE _holoscan_archive_output
    ERROR_VARIABLE _holoscan_archive_error
)
if(NOT _holoscan_archive_result EQUAL 0)
    message(FATAL_ERROR
        "FetchContent archive patch test failed:\n"
        "${_holoscan_archive_output}\n${_holoscan_archive_error}")
endif()
