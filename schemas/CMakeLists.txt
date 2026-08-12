# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(HOLOSCAN_SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(HOLOSCAN_SCHEMA_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated/holoscan/schemas")
set(HOLOSCAN_SCHEMA_FILES
    "${HOLOSCAN_SCHEMA_DIR}/header.fbs"
    "${HOLOSCAN_SCHEMA_DIR}/image.fbs"
    "${HOLOSCAN_SCHEMA_DIR}/imu.fbs"
    "${HOLOSCAN_SCHEMA_DIR}/string.fbs"
    "${HOLOSCAN_SCHEMA_DIR}/tensor.fbs"
    "${HOLOSCAN_SCHEMA_DIR}/tensor_packet.fbs"
    "${HOLOSCAN_SCHEMA_DIR}/time.fbs"
)

add_library(holoscan_fbs_tools INTERFACE)
add_library(${HOLOSCAN_PACKAGE_NAME}::fbs_tools ALIAS holoscan_fbs_tools)
set_target_properties(holoscan_fbs_tools PROPERTIES
    HOLOSCAN_FLATC_VERSION "${Flatbuffers_VERSION}"
    HOLOSCAN_SCHEMA_DIR "${HOLOSCAN_SCHEMA_DIR}"
    HOLOSCAN_FLATBUFFERS_REFLECTION_SCHEMA
        "${HOLOSCAN_SCHEMA_DIR}/flatbuffers/reflection.fbs"
)

holoscan_add_flatbuffer_schema(
    TARGET holoscan_flatbuffer_schemas_generated
    SCHEMAS ${HOLOSCAN_SCHEMA_FILES}
    INCLUDE_DIRS "${HOLOSCAN_SCHEMA_DIR}"
    OUTPUT_DIR "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
    BFBS_EMBED
)
set(HOLOSCAN_SCHEMA_GENERATED_HEADERS
    ${holoscan_flatbuffer_schemas_generated_GENERATED_HEADERS}
)

add_library(holoscan_schema
    "${HOLOSCAN_TOP}/src/schema/image_metadata.cpp"
    "${HOLOSCAN_TOP}/src/schema/image_port_type_ops.cpp"
    "${HOLOSCAN_TOP}/src/schema/tensor_overrides.cpp"
    "${HOLOSCAN_TOP}/src/schema/tensor_packet_port_type_ops.cpp"
)
add_library(${HOLOSCAN_PACKAGE_NAME}::schema ALIAS holoscan_schema)
add_dependencies(holoscan_schema holoscan_flatbuffer_schemas_generated)
holoscan_configure_cxx20_target(holoscan_schema PUBLIC)
target_include_directories(holoscan_schema
    PRIVATE
        "${HOLOSCAN_TOP}/src"
    PUBLIC
        $<BUILD_INTERFACE:${HOLOSCAN_TOP}/include>
        $<BUILD_INTERFACE:${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${Flatbuffers_SOURCE_DIR}/include>
        "$<BUILD_INTERFACE:${HOLOSCAN_SCHEMA_GENERATED_DIR}>"
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/holoscan/schemas>
)
target_link_libraries(holoscan_schema
    PUBLIC
        CUDA::cudart
        holoscan::core
    PRIVATE
        flatbuffers::flatbuffers
        fmt::fmt-header-only
)
target_compile_definitions(holoscan_schema
    PUBLIC $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:HOLOSCAN_SCHEMA_STATIC_DEFINE>
)
generate_export_header(holoscan_schema
    BASE_NAME HOLOSCAN_SCHEMA
    EXPORT_MACRO_NAME HOLOSCAN_SCHEMA_EXPORT
    NO_EXPORT_MACRO_NAME HOLOSCAN_SCHEMA_NO_EXPORT
    DEPRECATED_MACRO_NAME HOLOSCAN_SCHEMA_DEPRECATED
    STATIC_DEFINE HOLOSCAN_SCHEMA_STATIC_DEFINE
    INCLUDE_GUARD_NAME HOLOSCAN_SCHEMA_EXPORT_H_GENERATED
    EXPORT_FILE_NAME
        "${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/include/holoscan/holoscan_schema_export.h"
)
set_target_properties(holoscan_schema PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    EXPORT_NAME schema
    OUTPUT_NAME holoscan_schema
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    VISIBILITY_INLINES_HIDDEN YES
)
if(HOLOSCAN_ENABLE_CLANG_TIDY AND CMAKE_CXX_CLANG_TIDY)
    # FlatBuffers generates declarations with short parameter names such as _fbb and _o.
    set(_holoscan_schema_clang_tidy_checks
        "${CLANG_TIDY_CHECKS},-readability-inconsistent-declaration-parameter-name"
    )
    set_target_properties(holoscan_schema
        PROPERTIES
            CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY};--checks=${_holoscan_schema_clang_tidy_checks}"
    )
    unset(_holoscan_schema_clang_tidy_checks)
endif()
holoscan_configure_linux_shared_link(holoscan_schema)

install(FILES
    "${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/include/holoscan/holoscan_schema_export.h"
    DESTINATION include/holoscan
    COMPONENT holoscan-core
)
install(DIRECTORY "${HOLOSCAN_TOP}/include/holoscan/schema"
    DESTINATION include/holoscan
    COMPONENT holoscan-core
    FILES_MATCHING PATTERN "*.hpp"
)
install(FILES ${HOLOSCAN_SCHEMA_GENERATED_HEADERS}
    DESTINATION include/holoscan/schemas
    COMPONENT holoscan-core
)
install(FILES ${HOLOSCAN_SCHEMA_FILES}
    DESTINATION ${CMAKE_INSTALL_DATADIR}/holoscan/schemas
    COMPONENT holoscan-core
)
install(FILES "${HOLOSCAN_SCHEMA_DIR}/flatbuffers/reflection.fbs"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/holoscan/schemas/flatbuffers
    COMPONENT holoscan-core
)

# Keep existing in-tree test and benchmark consumers working until they consume target properties.
set(HOLOSCAN_SCHEMA_DIR "${HOLOSCAN_SCHEMA_DIR}" PARENT_SCOPE)
set(HOLOSCAN_SCHEMA_GENERATED_DIR "${HOLOSCAN_SCHEMA_GENERATED_DIR}" PARENT_SCOPE)
set(holoscan_SCHEMA_TOOLS_DIR "${HOLOSCAN_TOP}/cmake" PARENT_SCOPE)
