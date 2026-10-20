# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required IN ITEMS
        HOLOSCAN_PROVIDER_FLATC_EXECUTABLE
        HOLOSCAN_PROVIDER_FLATBUFFERS_INCLUDE_DIR)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

add_library(flatbuffers::flatbuffers INTERFACE IMPORTED GLOBAL)
set_target_properties(flatbuffers::flatbuffers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${HOLOSCAN_PROVIDER_FLATBUFFERS_INCLUDE_DIR}"
)
add_executable(flatbuffers::flatc IMPORTED GLOBAL)
set_target_properties(flatbuffers::flatc PROPERTIES
    IMPORTED_LOCATION "${HOLOSCAN_PROVIDER_FLATC_EXECUTABLE}"
)
set(Flatbuffers_VERSION 25.12.19)

function(add_provider_header_consumer)
    add_library(flatbuffers_provider_header_consumer OBJECT
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/header_consumer.cpp")
    target_link_libraries(flatbuffers_provider_header_consumer PRIVATE
        "$<COMPILE_ONLY:holoscan::schema>")
endfunction()
cmake_language(DEFER CALL add_provider_header_consumer)
