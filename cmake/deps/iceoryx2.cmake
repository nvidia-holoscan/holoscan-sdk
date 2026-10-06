# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# iceoryx2 exposes its Rust shared artifact through an INTERFACE wrapper rather
# than a SHARED target. Read the selected link input, not an SDK install prefix.
# This adapter intentionally handles the pinned Linux source/export contracts;
# it is not a recursive runtime dependency collector.
function(_holoscan_get_iceoryx2_ffi_library output)
    if(NOT TARGET iceoryx2-c::shared-lib)
        message(FATAL_ERROR "iceoryx2 must provide iceoryx2-c::shared-lib for wheel packaging")
    endif()
    get_target_property(link_inputs iceoryx2-c::shared-lib INTERFACE_LINK_LIBRARIES)
    set(artifacts "")
    foreach(input IN LISTS link_inputs)
        if(input MATCHES "^\\$<BUILD_INTERFACE:(.*)>$")
            set(input "${CMAKE_MATCH_1}")
        endif()
        if(IS_ABSOLUTE "${input}" AND input MATCHES "/libiceoryx2_ffi_c\\.so$")
            list(APPEND artifacts "${input}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES artifacts)
    list(LENGTH artifacts artifact_count)
    if(NOT artifact_count EQUAL 1)
        message(FATAL_ERROR
            "iceoryx2 must expose exactly one absolute Linux FFI shared-library link input; "
            "cannot select the wheel runtime from its package metadata")
    endif()
    # Source-build byproducts need not exist until the native build completes.
    set(${output} "${artifacts}" PARENT_SCOPE)
endfunction()

include(${rapids-cmake-dir}/cpm/find.cmake)

if(BUILD_SHARED_LIBS)
    set(_holomq_iceoryx2_cxx_target iceoryx2-cxx::shared-lib-cxx)
    set(_holomq_iceoryx2_cxx_unqualified_target shared-lib-cxx)
else()
    set(_holomq_iceoryx2_cxx_target iceoryx2-cxx::static-lib-cxx)
    set(_holomq_iceoryx2_cxx_unqualified_target static-lib-cxx)
endif()

if(TARGET ${_holomq_iceoryx2_cxx_target})
    unset(_holomq_iceoryx2_cxx_target)
    unset(_holomq_iceoryx2_cxx_unqualified_target)
    return()
endif()

set(HOLOMQ_ICEORYX2_CXX_VERSION "0.9.3" CACHE STRING "iceoryx2-cxx package version required by HoloMQ")
set(HOLOMQ_ICEORYX2_REPOSITORY "https://github.com/eclipse-iceoryx/iceoryx2.git")
set(HOLOMQ_ICEORYX2_GIT_TAG "6ef7c55176a8487ae7ca53c2e0649d5df9a1b3b4")
set(HOLOMQ_ICEORYX2_PATCH_DIR "${CMAKE_CURRENT_LIST_DIR}/patches/iceoryx2")
set(HOLOMQ_ICEORYX2_PATCHES
    "${HOLOMQ_ICEORYX2_PATCH_DIR}/0001-add-sample-released-handler.patch"
    "${HOLOMQ_ICEORYX2_PATCH_DIR}/0002-add-sample-replaced-handler.patch"
    "${HOLOMQ_ICEORYX2_PATCH_DIR}/0003-defer-blocking-send-wait-setup.patch"
    "${HOLOMQ_ICEORYX2_PATCH_DIR}/0004-name-library-install-component.patch"
    "${HOLOMQ_ICEORYX2_PATCH_DIR}/0005-clean-cargo-target-directory.patch"
    "${HOLOMQ_ICEORYX2_PATCH_DIR}/0006-accept-cargo-source-build.patch"
    "${HOLOMQ_ICEORYX2_PATCH_DIR}/0007-add-soname-to-iceoryx2-ffi.patch"
)

find_package(iceoryx2-cxx ${HOLOMQ_ICEORYX2_CXX_VERSION} CONFIG QUIET
    HINTS
        "${HOLOMQ_ICEORYX2_PREFIX}"
        "$ENV{HOLOMQ_ICEORYX2_PREFIX}"
    NO_DEFAULT_PATH
)

if(TARGET ${_holomq_iceoryx2_cxx_target})
    unset(_holomq_iceoryx2_cxx_target)
    unset(_holomq_iceoryx2_cxx_unqualified_target)
    return()
endif()

foreach(_holomq_iceoryx2_patch IN LISTS HOLOMQ_ICEORYX2_PATCHES)
    if(NOT EXISTS "${_holomq_iceoryx2_patch}")
        message(FATAL_ERROR "HoloMQ iceoryx2 patch not found: ${_holomq_iceoryx2_patch}")
    endif()
endforeach()
unset(_holomq_iceoryx2_patch)

find_program(CARGO_EXECUTABLE cargo)
if(NOT CARGO_EXECUTABLE)
    message(FATAL_ERROR
        "HoloMQ iceoryx2 backend requires either a prebuilt iceoryx2 install prefix or cargo "
        "for the from-source fallback. Set HOLOMQ_ICEORYX2_PREFIX to an install prefix that "
        "provides iceoryx2-cxx ${HOLOMQ_ICEORYX2_CXX_VERSION}, or install a Rust toolchain.")
endif()

holoscan_generate_dependency_patch_command(_holomq_iceoryx2_patch_command
    PATCHES ${HOLOMQ_ICEORYX2_PATCHES})

rapids_cpm_find(iceoryx2-cxx ${HOLOMQ_ICEORYX2_CXX_VERSION}
    GLOBAL_TARGETS
        ${_holomq_iceoryx2_cxx_target}
    CPM_ARGS
        GIT_REPOSITORY ${HOLOMQ_ICEORYX2_REPOSITORY}
        GIT_TAG ${HOLOMQ_ICEORYX2_GIT_TAG}
        GIT_SHALLOW FALSE
        PATCH_COMMAND ${_holomq_iceoryx2_patch_command}
        EXCLUDE_FROM_ALL OFF
        OVERRIDE_FIND_PACKAGE
    OPTIONS
        "BUILD_EXAMPLES OFF"
        "BUILD_TESTING OFF"
        "BUILD_CXX ON"
)
unset(_holomq_iceoryx2_patch_command)

if(NOT TARGET ${_holomq_iceoryx2_cxx_target} AND TARGET ${_holomq_iceoryx2_cxx_unqualified_target})
    add_library(${_holomq_iceoryx2_cxx_target} ALIAS ${_holomq_iceoryx2_cxx_unqualified_target})
endif()

if(NOT TARGET ${_holomq_iceoryx2_cxx_target})
    message(FATAL_ERROR
        "iceoryx2-cxx dependency must provide ${_holomq_iceoryx2_cxx_target} "
        "when BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}")
endif()

unset(_holomq_iceoryx2_cxx_target)
unset(_holomq_iceoryx2_cxx_unqualified_target)
