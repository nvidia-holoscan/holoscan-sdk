# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)
project(holoscan_flatbuffers_predeclared_provider LANGUAGES NONE)

foreach(required IN ITEMS
        HOLOSCAN_FLATC_EXECUTABLE
        HOLOSCAN_FLATBUFFERS_INCLUDE_DIR
        HOLOSCAN_FLATBUFFERS_DEPENDENCY_FILE)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

add_library(flatbuffers::flatbuffers INTERFACE IMPORTED GLOBAL)
set_target_properties(flatbuffers::flatbuffers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${HOLOSCAN_FLATBUFFERS_INCLUDE_DIR}"
)
add_executable(flatbuffers::flatc IMPORTED GLOBAL)
set_target_properties(flatbuffers::flatc PROPERTIES
    IMPORTED_LOCATION "${HOLOSCAN_FLATC_EXECUTABLE}"
)
set(Flatbuffers_VERSION 25.12.19)
set(rapids-cmake-dir "${CMAKE_CURRENT_SOURCE_DIR}/rapids")

function(load_holoscan_flatbuffers_dependency)
    include("${HOLOSCAN_FLATBUFFERS_DEPENDENCY_FILE}")
    set(holoscan_provider_flatbuffers_version "${Flatbuffers_VERSION}" PARENT_SCOPE)
endfunction()
load_holoscan_flatbuffers_dependency()

if(NOT holoscan_provider_flatbuffers_version VERSION_EQUAL "25.12.19")
    message(FATAL_ERROR
        "Predeclared provider did not propagate the exact configured FlatBuffers version"
    )
endif()
