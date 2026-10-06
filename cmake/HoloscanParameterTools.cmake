# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

option(HOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER
       "Build the experimental loopback Parameter tuner (default ON per HSDK5-019 X09)" ON)

# params and params_io are part of the holoscan-core component: they are separate DSOs that link
# core, config, and config_io one-way, and are installed through the shared holoscan-exports set.
if(NOT TARGET core OR NOT TARGET config OR NOT TARGET config_io)
    message(FATAL_ERROR
        "HoloscanParameterTools.cmake requires the core, config, and config_io targets")
endif()

add_library(params
    src/params/config.cpp
    src/params/error.cpp
    src/params/rebind.cpp
)
add_library(${HOLOSCAN_PACKAGE_NAME}::params ALIAS params)
holoscan_configure_cxx20_target(params PUBLIC)
holoscan_apply_logging_policy(params)
target_include_directories(params
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(params
    PUBLIC
        holoscan::config
        holoscan::core
)
target_compile_definitions(params
    PUBLIC $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:HOLOSCAN_PARAMS_STATIC_DEFINE>
)
generate_export_header(params
    BASE_NAME HOLOSCAN_PARAMS
    EXPORT_MACRO_NAME HOLOSCAN_PARAMS_EXPORT
    NO_EXPORT_MACRO_NAME HOLOSCAN_PARAMS_NO_EXPORT
    DEPRECATED_MACRO_NAME HOLOSCAN_PARAMS_DEPRECATED
    STATIC_DEFINE HOLOSCAN_PARAMS_STATIC_DEFINE
    INCLUDE_GUARD_NAME HOLOSCAN_PARAMS_EXPORT_H_GENERATED
    EXPORT_FILE_NAME
        "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_params_export.h"
)
set_target_properties(params PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    EXPORT_NAME params
    OUTPUT_NAME holoscan_params
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    VISIBILITY_INLINES_HIDDEN YES
)
holoscan_configure_linux_shared_link(params)
list(APPEND HOLOSCAN_INSTALL_TARGETS params)

add_library(params_io src/params/persistence_v1.cpp)
add_library(${HOLOSCAN_PACKAGE_NAME}::params_io ALIAS params_io)
holoscan_configure_cxx20_target(params_io PUBLIC)
holoscan_apply_logging_policy(params_io)
target_include_directories(params_io
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(params_io
    PUBLIC
        holoscan::config_io
        holoscan::params
)
target_compile_definitions(params_io
    PUBLIC $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:HOLOSCAN_PARAMS_IO_STATIC_DEFINE>
)
generate_export_header(params_io
    BASE_NAME HOLOSCAN_PARAMS_IO
    EXPORT_MACRO_NAME HOLOSCAN_PARAMS_IO_EXPORT
    NO_EXPORT_MACRO_NAME HOLOSCAN_PARAMS_IO_NO_EXPORT
    DEPRECATED_MACRO_NAME HOLOSCAN_PARAMS_IO_DEPRECATED
    STATIC_DEFINE HOLOSCAN_PARAMS_IO_STATIC_DEFINE
    INCLUDE_GUARD_NAME HOLOSCAN_PARAMS_IO_EXPORT_H_GENERATED
    EXPORT_FILE_NAME
        "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_params_io_export.h"
)
set_target_properties(params_io PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    EXPORT_NAME params_io
    OUTPUT_NAME holoscan_params_io
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    VISIBILITY_INLINES_HIDDEN YES
)
holoscan_configure_linux_shared_link(params_io)
list(APPEND HOLOSCAN_INSTALL_TARGETS params_io)

if(HOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER)
    # Keep the tuner's strict JSON grammar behind the same pinned, source-only yyjson
    # provider used by config_io, reusing its private objects.
    add_library(experimental_params_web_tuner
        src/params/web_tuner.cpp
        $<TARGET_OBJECTS:holoscan_config_yyjson_private_objects>
    )
    add_library(${HOLOSCAN_PACKAGE_NAME}::experimental_params_web_tuner
                ALIAS experimental_params_web_tuner)
    holoscan_configure_cxx20_target(experimental_params_web_tuner PUBLIC)
    holoscan_apply_logging_policy(experimental_params_web_tuner)
    target_include_directories(experimental_params_web_tuner
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_include_directories(experimental_params_web_tuner SYSTEM PRIVATE
        ${yyjson_SOURCE_DIR}/src
    )
    target_link_libraries(experimental_params_web_tuner
        PUBLIC holoscan::params
        PRIVATE Threads::Threads
    )
    target_compile_definitions(experimental_params_web_tuner
        PRIVATE yyjson_api=
        PUBLIC
            $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:
                HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_STATIC_DEFINE>
    )
    generate_export_header(experimental_params_web_tuner
        BASE_NAME HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER
        EXPORT_MACRO_NAME HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_EXPORT
        NO_EXPORT_MACRO_NAME HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_NO_EXPORT
        DEPRECATED_MACRO_NAME HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_DEPRECATED
        STATIC_DEFINE HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_STATIC_DEFINE
        INCLUDE_GUARD_NAME HOLOSCAN_EXPERIMENTAL_PARAMS_WEB_TUNER_EXPORT_H_GENERATED
        EXPORT_FILE_NAME
            "${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_experimental_params_web_tuner_export.h"
    )
    set_target_properties(experimental_params_web_tuner PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        EXPORT_NAME experimental_params_web_tuner
        OUTPUT_NAME holoscan_experimental_params_web_tuner
        POSITION_INDEPENDENT_CODE ON
        SOVERSION ${PROJECT_VERSION_MAJOR}
        VERSION ${PROJECT_VERSION}
        VISIBILITY_INLINES_HIDDEN YES
    )
    holoscan_configure_linux_shared_link(experimental_params_web_tuner)
    install(TARGETS experimental_params_web_tuner
        DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}
        EXPORT ${HOLOSCAN_PACKAGE_NAME}-experimental-params-web-tuner-exports
        COMPONENT holoscan-experimental-params-web-tuner
    )
    install(EXPORT ${HOLOSCAN_PACKAGE_NAME}-experimental-params-web-tuner-exports
        FILE ${HOLOSCAN_PACKAGE_NAME}-experimental-params-web-tuner-targets.cmake
        NAMESPACE ${HOLOSCAN_PACKAGE_NAME}::
        DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}/cmake/${HOLOSCAN_PACKAGE_NAME}
        COMPONENT holoscan-experimental-params-web-tuner
    )
    holoscan_install_tooling_metadata(
        experimental_params_web_tuner.json.in
        holoscan-experimental-params-web-tuner.json
        holoscan-experimental-params-web-tuner)
endif()

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_params_export.h
    ${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_params_io_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan
    COMPONENT holoscan-core
)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/params/config.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/params/error.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/params/persistence.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/params/register_operator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/params/rebind.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/params
    COMPONENT holoscan-core
)
if(HOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_experimental_params_web_tuner_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan
        COMPONENT holoscan-experimental-params-web-tuner
    )
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/experimental/params/web_tuner.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/experimental/params
        COMPONENT holoscan-experimental-params-web-tuner
    )
endif()
