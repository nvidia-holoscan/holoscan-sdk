# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(HOLOSCAN_SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(HOLOSCAN_SCHEMA_GENERATED_DIR "${holoscan_BINARY_DIR}/generated/holoscan/schemas")
set(HOLOSCAN_SCHEMA_FILES
    "${HOLOSCAN_SCHEMA_DIR}/geometry.fbs"
    "${HOLOSCAN_SCHEMA_DIR}/header.fbs"
    "${HOLOSCAN_SCHEMA_DIR}/string.fbs"
    "${HOLOSCAN_SCHEMA_DIR}/tensor.fbs"
    "${HOLOSCAN_SCHEMA_DIR}/tensor_packet.fbs"
    "${HOLOSCAN_SCHEMA_DIR}/time.fbs"
)
set(_HOLOSCAN_SCHEMA_PYTHON_OUTPUT_DIR
    "${HOLOSCAN_PYTHON_MODULE_ROOT}/holoscan/schema")
set(_HOLOSCAN_GEOMETRY_PYTHON_ARGS)
set(_HOLOSCAN_HEADER_PYTHON_ARGS)
set(_HOLOSCAN_ROOTED_SCHEMA_PYTHON_ARGS)
set(_HOLOSCAN_TENSOR_PACKET_PYTHON_ARGS)
if(HOLOSCAN_BUILD_PYTHON)
    set(_HOLOSCAN_CORE_SCHEMA_PYTHON_ARGS
        PYTHON_INSTALL_COMPONENT holoscan-python_libs
        PYTHON_OUTPUT_DIR "${_HOLOSCAN_SCHEMA_PYTHON_OUTPUT_DIR}")
    list(APPEND _HOLOSCAN_GEOMETRY_PYTHON_ARGS
        PYTHON_VALUES_ONLY
        ${_HOLOSCAN_CORE_SCHEMA_PYTHON_ARGS})
    list(APPEND _HOLOSCAN_HEADER_PYTHON_ARGS
        PYTHON
        PYTHON_CODEC_DEFERRED
        ${_HOLOSCAN_CORE_SCHEMA_PYTHON_ARGS}
        PYTHON_EXTERNAL_TYPES "holoscan.schema.Time=holoscan.schema.time")
    list(APPEND _HOLOSCAN_ROOTED_SCHEMA_PYTHON_ARGS
        PYTHON
        PYTHON_CODEC_DEFERRED
        ${_HOLOSCAN_CORE_SCHEMA_PYTHON_ARGS})
    list(APPEND _HOLOSCAN_TENSOR_PACKET_PYTHON_ARGS
        PYTHON
        PYTHON_CODEC_DEFERRED
        ${_HOLOSCAN_CORE_SCHEMA_PYTHON_ARGS})
endif()

add_library(holoscan_fbs_tools INTERFACE)
add_library(${HOLOSCAN_PACKAGE_NAME}::fbs_tools ALIAS holoscan_fbs_tools)
set_target_properties(holoscan_fbs_tools PROPERTIES
    HOLOSCAN_FLATC_VERSION "${Flatbuffers_VERSION}"
    HOLOSCAN_SCHEMA_DIR "${HOLOSCAN_SCHEMA_DIR}"
    HOLOSCAN_FLATBUFFERS_REFLECTION_SCHEMA
        "${HOLOSCAN_SCHEMA_DIR}/flatbuffers/reflection.fbs"
)

# Every SDK-owned schema that generates a binding also embeds its reflection bytes, so a consumer
# that must describe a payload to a reader compiled against no schema at all -- a recorder writing
# a self-describing log, for instance -- can reach holoscan::binary_schema<T>() for any built-in
# payload type rather than regenerating the .fbs itself and risking a second identity for the same
# wire family. The bytes are diagnostic provenance only and never enter identity or the wire. They
# live in an inline function, so a consumer that never calls the accessor pays nothing in its
# binary. geometry.fbs and tensor.fbs are omitted because neither generates a binding to attach
# them to: geometry contributes field types rather than a root, and Tensor's binding is written by
# hand in flatbuffer_schema_package.hpp.
holoscan_add_flatbuffer_schema(
    TARGET holoscan_geometry_schema_generated
    SCHEMA geometry.fbs
    OUTPUT_DIR "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
    ${_HOLOSCAN_GEOMETRY_PYTHON_ARGS}
)
holoscan_add_flatbuffer_schema(
    TARGET holoscan_tensor_schema_generated
    SCHEMA tensor.fbs
    OUTPUT_DIR "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
)
holoscan_add_flatbuffer_schema(
    TARGET holoscan_header_schema_generated
    SCHEMA header.fbs
    OUTPUT_DIR "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
    MAX_SERIALIZED_SIZE 512
    BFBS_EMBED
    ${_HOLOSCAN_HEADER_PYTHON_ARGS}
)
holoscan_add_flatbuffer_schema(
    TARGET holoscan_string_schema_generated
    SCHEMA string.fbs
    OUTPUT_DIR "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
    MAX_SERIALIZED_SIZE 512
    BFBS_EMBED
    ${_HOLOSCAN_ROOTED_SCHEMA_PYTHON_ARGS}
)
holoscan_add_flatbuffer_schema(
    TARGET holoscan_tensor_packet_schema_generated
    SCHEMA tensor_packet.fbs
    OUTPUT_DIR "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
    MAX_SERIALIZED_SIZE 512
    BFBS_EMBED
    ${_HOLOSCAN_TENSOR_PACKET_PYTHON_ARGS}
)
holoscan_add_flatbuffer_schema(
    TARGET holoscan_time_schema_generated
    SCHEMA time.fbs
    OUTPUT_DIR "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
    MAX_SERIALIZED_SIZE 512
    BFBS_EMBED
    ${_HOLOSCAN_ROOTED_SCHEMA_PYTHON_ARGS}
)

if(HOLOSCAN_BUILD_PYTHON)
    file(RELATIVE_PATH _holoscan_schema_python_build_lib_relative
        "${_HOLOSCAN_SCHEMA_PYTHON_OUTPUT_DIR}"
        "${holoscan_BINARY_DIR}/lib")
    set(_holoscan_schema_python_install_dir
        "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/holoscan/schema")
    file(RELATIVE_PATH _holoscan_schema_python_install_lib_relative
        "/${_holoscan_schema_python_install_dir}"
        "/${CMAKE_INSTALL_LIBDIR}")
    get_property(_holoscan_schema_python_targets GLOBAL
        PROPERTY HOLOSCAN_PYTHON_SCHEMA_INSTALL_TARGETS)
    foreach(_holoscan_schema_target IN LISTS _holoscan_schema_python_targets)
        get_target_property(_holoscan_schema_python_component
            "${_holoscan_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_INSTALL_COMPONENT)
        if(NOT _holoscan_schema_python_component STREQUAL "holoscan-python_libs")
            continue()
        endif()
        get_target_property(_holoscan_schema_python_codec_target
            "${_holoscan_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_CODEC_TARGET)
        if(NOT _holoscan_schema_python_codec_target OR
           _holoscan_schema_python_codec_target MATCHES "-NOTFOUND$")
            continue()
        endif()
        set_target_properties("${_holoscan_schema_python_codec_target}" PROPERTIES
            BUILD_RPATH "\$ORIGIN/${_holoscan_schema_python_build_lib_relative}"
            INSTALL_RPATH "\$ORIGIN/${_holoscan_schema_python_install_lib_relative}")
    endforeach()
    unset(_holoscan_schema_python_build_lib_relative)
    unset(_holoscan_schema_python_codec_target)
    unset(_holoscan_schema_python_component)
    unset(_holoscan_schema_python_install_dir)
    unset(_holoscan_schema_python_install_lib_relative)
    unset(_holoscan_schema_target)
    unset(_holoscan_schema_python_targets)
endif()
unset(_HOLOSCAN_CORE_SCHEMA_PYTHON_ARGS)
unset(_HOLOSCAN_GEOMETRY_PYTHON_ARGS)
unset(_HOLOSCAN_HEADER_PYTHON_ARGS)
unset(_HOLOSCAN_ROOTED_SCHEMA_PYTHON_ARGS)
unset(_HOLOSCAN_SCHEMA_PYTHON_OUTPUT_DIR)
unset(_HOLOSCAN_TENSOR_PACKET_PYTHON_ARGS)
add_custom_target(holoscan_flatbuffer_schemas_generated)
add_dependencies(holoscan_flatbuffer_schemas_generated
    holoscan_geometry_schema_generated
    holoscan_header_schema_generated
    holoscan_string_schema_generated
    holoscan_tensor_packet_schema_generated
    holoscan_tensor_schema_generated
    holoscan_time_schema_generated
)
set(HOLOSCAN_SCHEMA_GENERATED_HEADERS
)
foreach(schema_target IN ITEMS
        holoscan_geometry_schema_generated
        holoscan_header_schema_generated
        holoscan_string_schema_generated
        holoscan_tensor_packet_schema_generated
        holoscan_tensor_schema_generated
        holoscan_time_schema_generated)
    get_target_property(schema_headers
        "${schema_target}" HOLOSCAN_FLATBUFFER_GENERATED_HEADERS)
    list(APPEND HOLOSCAN_SCHEMA_GENERATED_HEADERS ${schema_headers})
endforeach()

add_library(holoscan_schema
    "${holoscan_SOURCE_DIR}/src/schema/authoring_registry.cpp"
    "${holoscan_SOURCE_DIR}/src/schema/schema_package.cpp"
    "${holoscan_SOURCE_DIR}/src/schema/tensor_overrides.cpp"
)
add_library(${HOLOSCAN_PACKAGE_NAME}::schema ALIAS holoscan_schema)
add_dependencies(holoscan_schema holoscan_flatbuffer_schemas_generated)
if(HOLOSCAN_BUILD_PYTHON)
    # These codecs are declared before holoscan_schema exists, so the schema helper cannot attach
    # its normal link automatically.  Link every rooted Core codec once the library is available;
    # values-only geometry and the hand-authored Tensor facade do not have codec targets.
    foreach(_holoscan_schema_python_target IN ITEMS
            holoscan_header_schema_generated
            holoscan_string_schema_generated
            holoscan_tensor_packet_schema_generated
            holoscan_time_schema_generated)
        get_target_property(_holoscan_schema_python_codec_target
            "${_holoscan_schema_python_target}" HOLOSCAN_FLATBUFFER_PYTHON_CODEC_TARGET)
        target_link_libraries("${_holoscan_schema_python_codec_target}"
            PRIVATE holoscan_schema)
    endforeach()
    unset(_holoscan_schema_python_codec_target)
    unset(_holoscan_schema_python_target)
endif()
holoscan_configure_cxx20_target(holoscan_schema PUBLIC)
target_include_directories(holoscan_schema
    PRIVATE
        "${holoscan_SOURCE_DIR}/src"
    PUBLIC
        $<BUILD_INTERFACE:${holoscan_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${holoscan_BINARY_DIR}/include>
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
        "${holoscan_BINARY_DIR}/include/holoscan/holoscan_schema_export.h"
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
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan
    COMPONENT holoscan-core
)
install(DIRECTORY "${holoscan_SOURCE_DIR}/include/holoscan/schema"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan
    COMPONENT holoscan-core
    FILES_MATCHING PATTERN "*.hpp"
)
install(FILES ${HOLOSCAN_SCHEMA_GENERATED_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/schemas
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
set(holoscan_SCHEMA_TOOLS_DIR "${holoscan_SOURCE_DIR}/cmake" PARENT_SCOPE)
