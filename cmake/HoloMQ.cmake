# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# HoloMQ is an SDK-owned component. Its public target remains independently
# linkable, but all source, packaging, and dependency policy comes from the
# enclosing Holoscan project.
set(HOLOMQ_SRC "${holoscan_SOURCE_DIR}/src/holomq")

# Map SDK backend cache names (HOLOMQ_DEFAULT_BACKEND / HOLOSCAN_MQ_BACKEND) to
# the upstream compile-time transport identifier.
if(NOT DEFINED HOLOMQ_TRANSPORT)
    if(NOT HOLOMQ_DEFAULT_BACKEND STREQUAL "iceoryx2")
        message(FATAL_ERROR
            "HOLOMQ_DEFAULT_BACKEND must be iceoryx2 (got \"${HOLOMQ_DEFAULT_BACKEND}\")")
    endif()
    set(HOLOMQ_TRANSPORT "iox2" CACHE STRING "Messaging backend: iox2")
endif()
set_property(CACHE HOLOMQ_TRANSPORT PROPERTY STRINGS iox2)

if(HOLOMQ_TRANSPORT STREQUAL "iox2")
    set(HOLOMQ_TRANSPORT_UPPER "IOX2")
else()
    message(FATAL_ERROR "HOLOMQ_TRANSPORT must be iox2 (got \"${HOLOMQ_TRANSPORT}\")")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(HOLOMQ_FD_PASS_SOCKET_DEFAULT "ABSTRACT")
else()
    set(HOLOMQ_FD_PASS_SOCKET_DEFAULT "PATH")
endif()
set(HOLOMQ_FD_PASS_SOCKET
    "${HOLOMQ_FD_PASS_SOCKET_DEFAULT}"
    CACHE STRING
        "fd_pass AF_UNIX endpoint: PATH (filesystem) or ABSTRACT (Linux abstract namespace; default ABSTRACT on Linux)")
set_property(CACHE HOLOMQ_FD_PASS_SOCKET PROPERTY STRINGS PATH ABSTRACT)
if(NOT HOLOMQ_FD_PASS_SOCKET STREQUAL "PATH" AND NOT HOLOMQ_FD_PASS_SOCKET STREQUAL "ABSTRACT")
    message(FATAL_ERROR "HOLOMQ_FD_PASS_SOCKET must be PATH or ABSTRACT (got \"${HOLOMQ_FD_PASS_SOCKET}\")")
endif()
if(HOLOMQ_FD_PASS_SOCKET STREQUAL "ABSTRACT" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "HOLOMQ_FD_PASS_SOCKET=ABSTRACT is only supported on Linux (got ${CMAKE_SYSTEM_NAME})")
endif()

function(holomq_apply_fd_pass_socket_mode target_name)
    if(HOLOMQ_FD_PASS_SOCKET STREQUAL "ABSTRACT")
        target_compile_definitions(${target_name} PUBLIC HOLOMQ_FD_PASS_USE_ABSTRACT=1)
    else()
        target_compile_definitions(${target_name} PUBLIC HOLOMQ_FD_PASS_USE_ABSTRACT=0)
    endif()
endfunction()

set(HOLOMQ_LIBRARY_SOURCES
    "${HOLOMQ_SRC}/buffer_pool/buffer.cpp"
    "${HOLOMQ_SRC}/buffer_pool/buffer_impl.cpp"
    "${HOLOMQ_SRC}/buffer_pool/const_buffer.cpp"
    "${HOLOMQ_SRC}/buffer_pool/const_buffer_impl.cpp"
    "${HOLOMQ_SRC}/buffer_pool/buffer_handle.cpp"
    "${HOLOMQ_SRC}/buffer_pool/buffer_pool.cpp"
    "${HOLOMQ_SRC}/buffer_pool/buffer_pool_impl.cpp"
    "${HOLOMQ_SRC}/buffer_pool/generation_slot_array.cpp"
    "${HOLOMQ_SRC}/buffer_pool/device/device_buffer_impl.cpp"
    "${HOLOMQ_SRC}/buffer_pool/device/device_buffer_pool_impl.cpp"
    "${HOLOMQ_SRC}/buffer_pool/device/default_device_buffer_allocator.cpp"
    "${HOLOMQ_SRC}/buffer_pool/device/vmm_exported_pool.cpp"
    "${HOLOMQ_SRC}/buffer_pool/device/device_const_buffer_impl.cpp"
    "${HOLOMQ_SRC}/buffer_pool/host/default_host_buffer_allocator.cpp"
    "${HOLOMQ_SRC}/buffer_pool/host/host_backing_segment_cache.cpp"
    "${HOLOMQ_SRC}/buffer_pool/host/host_buffer_impl.cpp"
    "${HOLOMQ_SRC}/buffer_pool/host/host_buffer_pool_impl.cpp"
    "${HOLOMQ_SRC}/buffer_pool/host/host_const_buffer_impl.cpp"
    "${HOLOMQ_SRC}/buffer_pool/host/shm_host_buffer_pool_memory.cpp"
    "${HOLOMQ_SRC}/buffer_pool/ref_count/ref_count_key.cpp"
    "${HOLOMQ_SRC}/buffer_pool/ref_count/shared_memory.cpp"
    "${HOLOMQ_SRC}/buffer_pool/ref_count/shm_ref_count.cpp"
    "${HOLOMQ_SRC}/buffer_pool/ref_count/ref_count_segment_cache.cpp"
    "${HOLOMQ_SRC}/buffer_pool/sync/cuda_stream_generation.cpp"
    "${HOLOMQ_SRC}/buffer_pool/sync/deferred_ref_release.cpp"
    "${HOLOMQ_SRC}/logger.cpp"
    "${HOLOMQ_SRC}/errors.cpp"
    "${HOLOMQ_SRC}/pubsub_capabilities.cpp"
    "${HOLOMQ_SRC}/context.cpp"
    "${HOLOMQ_SRC}/context_impl.cpp"
    "${HOLOMQ_SRC}/context_builder.cpp"
    "${HOLOMQ_SRC}/vmm_pool_export_registry.cpp"
    "${HOLOMQ_SRC}/cuda/mem.cpp"
    "${HOLOMQ_SRC}/cuda/host_registration.cpp"
    "${HOLOMQ_SRC}/cuda/mem_mapping.cpp"
    "${HOLOMQ_SRC}/cuda/mem_address_reservation.cpp"
    "${HOLOMQ_SRC}/vmm_pool_import_cache.cpp"
    "${HOLOMQ_SRC}/cuda/context.cpp"
    "${HOLOMQ_SRC}/cuda/current_device.cpp"
    "${HOLOMQ_SRC}/cuda/stream.cpp"
    "${HOLOMQ_SRC}/fd_pass/client.cpp"
    "${HOLOMQ_SRC}/fd_pass/server.cpp"
    "${HOLOMQ_SRC}/fd_pass/sockaddr.cpp"
    "${HOLOMQ_SRC}/wait_set.cpp"
    "${HOLOMQ_SRC}/wait_trigger.cpp"
)

list(APPEND HOLOMQ_LIBRARY_SOURCES "${HOLOMQ_SRC}/transport/iox2/runtime.cpp")
list(APPEND HOLOMQ_LIBRARY_SOURCES "${HOLOMQ_SRC}/transport/iox2/wait_set_backend.cpp")
list(APPEND HOLOMQ_LIBRARY_SOURCES "${HOLOMQ_SRC}/transport/iox2/log_bridge.cpp")

function(holomq_apply_public_interface target_name)
    target_compile_features(${target_name} PUBLIC cxx_std_20)
    target_include_directories(${target_name}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            $<BUILD_INTERFACE:${HOLOMQ_SRC}>
    )
    target_compile_definitions(${target_name}
        PUBLIC
            "HOLOMQ_TRANSPORT_${HOLOMQ_TRANSPORT_UPPER}=1"
            $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:HOLOMQ_STATIC_DEFINE>
    )
    holomq_apply_fd_pass_socket_mode(${target_name})
    target_link_libraries(${target_name} PUBLIC CUDA::cuda_driver ${CMAKE_DL_LIBS})
    target_link_libraries(${target_name} PUBLIC tl::expected)
    target_link_libraries(${target_name} PRIVATE magic_enum::magic_enum)
    # Typed HoloMQ methods are instantiated in consumers. A shared HoloMQ build must use the
    # shared iceoryx2 binding so the DSO and those consumer instantiations share one process-wide
    # Rust FFI instance. Embedding a hidden static FFI copy in each binary breaks endpoint state.
    if(BUILD_SHARED_LIBS)
        target_link_libraries(${target_name} PUBLIC iceoryx2-cxx::shared-lib-cxx)
    else()
        target_link_libraries(${target_name} PUBLIC iceoryx2-cxx::static-lib-cxx)
    endif()
endfunction()

# Compile once into an OBJECT library, then link the configured SDK artifact.
add_library(holomq_objects OBJECT ${HOLOMQ_LIBRARY_SOURCES})
set_target_properties(holomq_objects PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
    VISIBILITY_INLINES_HIDDEN YES
)

find_package(iceoryx2-cxx REQUIRED QUIET)
find_package(tl-expected REQUIRED CONFIG QUIET)
find_package(magic_enum REQUIRED CONFIG QUIET)

holomq_apply_public_interface(holomq_objects)

if(BUILD_SHARED_LIBS)
    add_library(holomq SHARED $<TARGET_OBJECTS:holomq_objects>)
    holomq_apply_public_interface(holomq)
else()
    add_library(holomq STATIC $<TARGET_OBJECTS:holomq_objects>)
    holomq_apply_public_interface(holomq)
endif()
add_library(holomq::holomq ALIAS holomq)
add_library(${HOLOSCAN_PACKAGE_NAME}::holomq ALIAS holomq)

message(STATUS "HoloMQ transport backend: ${HOLOMQ_TRANSPORT}")
message(STATUS "HoloMQ FD-pass socket mode: ${HOLOMQ_FD_PASS_SOCKET}")

generate_export_header(holomq
    BASE_NAME HOLOMQ
    EXPORT_MACRO_NAME HOLOMQ_EXPORT
    NO_EXPORT_MACRO_NAME HOLOMQ_NO_EXPORT
    DEPRECATED_MACRO_NAME HOLOMQ_DEPRECATED
    STATIC_DEFINE HOLOMQ_STATIC_DEFINE
    INCLUDE_GUARD_NAME HOLOMQ_EXPORT_H_GENERATED
    EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/include/holomq/holomq_export.h"
)
set_target_properties(holomq PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    EXPORT_NAME holomq
    OUTPUT_NAME holomq
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    VISIBILITY_INLINES_HIDDEN YES
)

if(BUILD_SHARED_LIBS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(holomq PRIVATE
        LINKER:--exclude-libs,ALL
        LINKER:--no-undefined
        LINKER:--gc-sections
    )
endif()

set(_holomq_export_semantics_script
    "${holoscan_SOURCE_DIR}/scripts/check_holomq_export_macro_semantics.py"
)
if(Python3_Interpreter_FOUND)
    add_custom_target(holomq-check-export-macro-semantics
        COMMAND ${Python3_EXECUTABLE}
            "${_holomq_export_semantics_script}"
            --root "${CMAKE_CURRENT_SOURCE_DIR}"
            --compile-commands "${CMAKE_BINARY_DIR}/compile_commands.json"
        WORKING_DIRECTORY "${holoscan_SOURCE_DIR}"
        COMMENT "Checking HoloMQ public-header export macro semantics"
        VERBATIM
    )
endif()
unset(_holomq_export_semantics_script)

set(_holomq_symbol_check_script
    "${holoscan_SOURCE_DIR}/scripts/check_holomq_symbol_baseline.sh"
)
set(_holomq_symbol_baseline
    "${holoscan_SOURCE_DIR}/ci/abi/holomq.exports.baseline.txt"
)
if(BUILD_SHARED_LIBS
    AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
    AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(NOT EXISTS "${_holomq_symbol_check_script}")
        message(FATAL_ERROR
            "HoloMQ ABI check script is missing: ${_holomq_symbol_check_script}")
    endif()
    if(NOT EXISTS "${_holomq_symbol_baseline}")
        message(FATAL_ERROR
            "HoloMQ ABI symbol baseline is missing: ${_holomq_symbol_baseline}")
    endif()
    add_custom_target(holomq-check-symbols
        COMMAND bash
            "${_holomq_symbol_check_script}"
            "$<TARGET_FILE:holomq>"
            "${_holomq_symbol_baseline}"
        DEPENDS holomq
        COMMENT "Checking HoloMQ exported symbol baseline"
        VERBATIM
    )
endif()
unset(_holomq_symbol_check_script)
unset(_holomq_symbol_baseline)

list(APPEND HOLOSCAN_INSTALL_TARGETS holomq)
