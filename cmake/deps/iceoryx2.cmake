# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

include(${rapids-cmake-dir}/cpm/find.cmake)

set(_holomq_iceoryx2_expected_variant "holomq-private-libraries-v1")

if(BUILD_SHARED_LIBS)
    set(_holomq_iceoryx2_cxx_target iceoryx2-cxx::shared-lib-cxx)
    set(_holomq_iceoryx2_cxx_unqualified_target shared-lib-cxx)
else()
    set(_holomq_iceoryx2_cxx_target iceoryx2-cxx::static-lib-cxx)
    set(_holomq_iceoryx2_cxx_unqualified_target static-lib-cxx)
endif()

if(TARGET ${_holomq_iceoryx2_cxx_target})
    if(NOT HOLOMQ_ICEORYX2_PACKAGE_VARIANT STREQUAL _holomq_iceoryx2_expected_variant)
        message(FATAL_ERROR
            "HoloMQ requires its patched iceoryx2 package variant "
            "'${_holomq_iceoryx2_expected_variant}', but the existing iceoryx2-cxx targets "
            "do not provide it. Remove the unpatched package or configure "
            "HOLOMQ_ICEORYX2_PREFIX with the HoloMQ-patched installation."
        )
    endif()
    unset(_holomq_iceoryx2_cxx_target)
    unset(_holomq_iceoryx2_cxx_unqualified_target)
    unset(_holomq_iceoryx2_expected_variant)
    return()
endif()

set(HOLOMQ_ICEORYX2_CXX_VERSION "0.8.999" CACHE STRING "iceoryx2-cxx package version required by HoloMQ")
set(HOLOMQ_ICEORYX2_REPOSITORY "https://github.com/eclipse-iceoryx/iceoryx2.git")
set(HOLOMQ_ICEORYX2_GIT_TAG "c837bf441419b3b0e7b295a02b0937f35aea5f7f")
set(HOLOMQ_ICEORYX2_PATCH_DIR "${CMAKE_CURRENT_LIST_DIR}/patches/iceoryx2")
set(HOLOMQ_ICEORYX2_PATCHES
    "${HOLOMQ_ICEORYX2_PATCH_DIR}/0001-add-sample-released-handler.patch"
    "${HOLOMQ_ICEORYX2_PATCH_DIR}/0002-add-sample-replaced-handler.patch"
    "${HOLOMQ_ICEORYX2_PATCH_DIR}/0003-configure-library-install-component.patch"
    "${HOLOMQ_ICEORYX2_PATCH_DIR}/0004-accept-cargo-source-build.patch"
)

find_package(iceoryx2-cxx ${HOLOMQ_ICEORYX2_CXX_VERSION} CONFIG QUIET
    HINTS
        "${HOLOMQ_ICEORYX2_PREFIX}"
        "$ENV{HOLOMQ_ICEORYX2_PREFIX}"
    NO_DEFAULT_PATH
)

if(TARGET ${_holomq_iceoryx2_cxx_target})
    if(NOT HOLOMQ_ICEORYX2_PACKAGE_VARIANT STREQUAL _holomq_iceoryx2_expected_variant)
        message(FATAL_ERROR
            "The iceoryx2-cxx package found for HoloMQ does not identify itself as the "
            "required patched variant '${_holomq_iceoryx2_expected_variant}'. Rebuild the "
            "HoloMQ iceoryx2 dependency or set HOLOMQ_ICEORYX2_PREFIX to a compatible prefix."
        )
    endif()
    unset(_holomq_iceoryx2_cxx_target)
    unset(_holomq_iceoryx2_cxx_unqualified_target)
    unset(_holomq_iceoryx2_expected_variant)
    return()
endif()

foreach(_holomq_iceoryx2_patch IN LISTS HOLOMQ_ICEORYX2_PATCHES)
    if(NOT EXISTS "${_holomq_iceoryx2_patch}")
        message(FATAL_ERROR "HoloMQ iceoryx2 patch not found: ${_holomq_iceoryx2_patch}")
    endif()
endforeach()
unset(_holomq_iceoryx2_patch)

find_program(GIT_EXECUTABLE git REQUIRED)
find_program(CARGO_EXECUTABLE cargo)
if(NOT CARGO_EXECUTABLE)
    message(FATAL_ERROR
        "HoloMQ iceoryx2 backend requires either a prebuilt iceoryx2 install prefix or cargo "
        "for the from-source fallback. Set HOLOMQ_ICEORYX2_PREFIX to an install prefix that "
        "provides iceoryx2-cxx ${HOLOMQ_ICEORYX2_CXX_VERSION}, or install a Rust toolchain.")
endif()

set(_holomq_iceoryx2_patch_script "${CMAKE_CURRENT_LIST_DIR}/apply_patch_checked.sh")
if(NOT EXISTS "${_holomq_iceoryx2_patch_script}")
    message(FATAL_ERROR "HoloMQ iceoryx2 patch helper not found: ${_holomq_iceoryx2_patch_script}")
endif()

set(HOLOMQ_ICEORYX2_LIBRARY_COMPONENT "${HOLOSCAN_DEPEND_INSTALL_COMPONENT}"
        CACHE STRING "Install component for HoloMQ's iceoryx2 library artifacts"
)
rapids_cpm_find(iceoryx2-cxx ${HOLOMQ_ICEORYX2_CXX_VERSION}
    GLOBAL_TARGETS
        ${_holomq_iceoryx2_cxx_target}
    CPM_ARGS
        GIT_REPOSITORY ${HOLOMQ_ICEORYX2_REPOSITORY}
        GIT_TAG ${HOLOMQ_ICEORYX2_GIT_TAG}
        GIT_SHALLOW FALSE
        PATCH_COMMAND bash "${_holomq_iceoryx2_patch_script}" "${GIT_EXECUTABLE}" ${HOLOMQ_ICEORYX2_PATCHES}
        EXCLUDE_FROM_ALL OFF
        OVERRIDE_FIND_PACKAGE
    OPTIONS
        "BUILD_EXAMPLES OFF"
        "BUILD_TESTING OFF"
        "BUILD_CXX ON"
        "HOLOMQ_ICEORYX2_LIBRARY_COMPONENT ${HOLOSCAN_DEPEND_INSTALL_COMPONENT}"
)
unset(_holomq_iceoryx2_patch_script)

if(NOT TARGET ${_holomq_iceoryx2_cxx_target} AND TARGET ${_holomq_iceoryx2_cxx_unqualified_target})
    add_library(${_holomq_iceoryx2_cxx_target} ALIAS ${_holomq_iceoryx2_cxx_unqualified_target})
endif()

if(NOT TARGET ${_holomq_iceoryx2_cxx_target})
    message(FATAL_ERROR
        "iceoryx2-cxx dependency must provide ${_holomq_iceoryx2_cxx_target} "
        "when BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}")
endif()

set(HOLOMQ_ICEORYX2_PACKAGE_VARIANT "${_holomq_iceoryx2_expected_variant}")
unset(_holomq_iceoryx2_cxx_target)
unset(_holomq_iceoryx2_cxx_unqualified_target)
unset(_holomq_iceoryx2_expected_variant)
