# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

option(HOLOSCAN_BUILD_PARAMS
       "Build the optional Holoscan Parameter binding and rebind library" ON)
option(HOLOSCAN_BUILD_PARAMS_IO
       "Build the optional Holoscan Parameter snapshot persistence library" ON)
option(HOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER
       "Build the experimental loopback Parameter tuner (default ON per HSDK5-019 X09)" ON)

# Because the tuner defaults ON, a params opt-out must disable the tuner explicitly as well.
if(HOLOSCAN_BUILD_PARAMS AND (NOT TARGET core OR NOT TARGET config))
    message(FATAL_ERROR
        "HOLOSCAN_BUILD_PARAMS requires the already-enabled core and config components")
endif()
if(HOLOSCAN_BUILD_PARAMS_IO AND NOT HOLOSCAN_BUILD_PARAMS)
    message(FATAL_ERROR
        "HOLOSCAN_BUILD_PARAMS_IO requires HOLOSCAN_BUILD_PARAMS=ON; dependencies are not auto-enabled")
endif()
if(HOLOSCAN_BUILD_PARAMS_IO AND NOT TARGET config_io)
    message(FATAL_ERROR
        "HOLOSCAN_BUILD_PARAMS_IO requires the config_io component")
endif()
if(HOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER AND NOT HOLOSCAN_BUILD_PARAMS)
    message(FATAL_ERROR
        "HOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER requires HOLOSCAN_BUILD_PARAMS=ON; "
        "set HOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER=OFF when opting out of params")
endif()

if(HOLOSCAN_BUILD_PARAMS)
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
    install(TARGETS params
        DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}
        EXPORT ${HOLOSCAN_PACKAGE_NAME}-params-exports
        COMPONENT holoscan-params
    )
    install(EXPORT ${HOLOSCAN_PACKAGE_NAME}-params-exports
        FILE ${HOLOSCAN_PACKAGE_NAME}-params-targets.cmake
        NAMESPACE ${HOLOSCAN_PACKAGE_NAME}::
        DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}/cmake/${HOLOSCAN_PACKAGE_NAME}
        COMPONENT holoscan-params
    )
    holoscan_install_tooling_metadata(params.json.in holoscan-params.json holoscan-params)

    if(HOLOSCAN_BUILD_PARAMS_IO)
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
    install(TARGETS params_io
        DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}
        EXPORT ${HOLOSCAN_PACKAGE_NAME}-params-io-exports
        COMPONENT holoscan-params-io
    )
    install(EXPORT ${HOLOSCAN_PACKAGE_NAME}-params-io-exports
        FILE ${HOLOSCAN_PACKAGE_NAME}-params-io-targets.cmake
        NAMESPACE ${HOLOSCAN_PACKAGE_NAME}::
        DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}/cmake/${HOLOSCAN_PACKAGE_NAME}
        COMPONENT holoscan-params-io
    )
    holoscan_install_tooling_metadata(
        params_io.json.in holoscan-params-io.json holoscan-params-io)
    endif()

    if(HOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER)
        # Keep the tuner's strict JSON grammar behind the same pinned, source-only yyjson
        # provider used by config_io. Reuse its private objects when config_io is enabled; the
        # fallback preserves the tuner's documented independence from the config_io component.
        if(TARGET holoscan_config_yyjson_private_objects)
            set(_holoscan_tuner_yyjson_objects holoscan_config_yyjson_private_objects)
        else()
            holoscan_add_dependency(yyjson)
            add_library(holoscan_tuner_yyjson_private_objects OBJECT
                "${yyjson_SOURCE_DIR}/src/yyjson.c"
            )
            target_include_directories(holoscan_tuner_yyjson_private_objects
                SYSTEM PRIVATE "${yyjson_SOURCE_DIR}/src"
            )
            target_compile_definitions(holoscan_tuner_yyjson_private_objects PRIVATE yyjson_api=)
            set_target_properties(holoscan_tuner_yyjson_private_objects PROPERTIES
                C_CLANG_TIDY ""
                C_EXTENSIONS NO
                C_STANDARD 11
                C_STANDARD_REQUIRED YES
                C_VISIBILITY_PRESET hidden
                POSITION_INDEPENDENT_CODE ON
            )
            set(_holoscan_tuner_yyjson_objects holoscan_tuner_yyjson_private_objects)
        endif()
        add_library(experimental_params_web_tuner
            src/params/web_tuner.cpp
            $<TARGET_OBJECTS:${_holoscan_tuner_yyjson_objects}>
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
        unset(_holoscan_tuner_yyjson_objects)
    endif()

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_params_export.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan
        COMPONENT holoscan-params
    )
    if(HOLOSCAN_BUILD_PARAMS_IO)
        install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_params_io_export.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan
            COMPONENT holoscan-params-io
        )
    endif()
    if(HOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER)
        install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/include/holoscan/holoscan_experimental_params_web_tuner_export.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan
            COMPONENT holoscan-experimental-params-web-tuner
        )
    endif()
    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/params/config.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/params/error.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/params/register_operator.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/params/rebind.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/params
        COMPONENT holoscan-params
    )
    if(HOLOSCAN_BUILD_PARAMS_IO)
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/params/persistence.hpp
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/params
            COMPONENT holoscan-params-io
        )
    endif()
    if(HOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER)
        install(FILES
            ${CMAKE_CURRENT_SOURCE_DIR}/include/holoscan/experimental/params/web_tuner.hpp
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/experimental/params
            COMPONENT holoscan-experimental-params-web-tuner
        )
    endif()
endif()
