# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

project(holoscan_compile_installed_consumer LANGUAGES CXX)

# REQ: CHAN-077

if(NOT DEFINED HOLOSCAN_TEST_INSTALL_PREFIX OR HOLOSCAN_TEST_INSTALL_PREFIX STREQUAL "")
    message(FATAL_ERROR "HOLOSCAN_TEST_INSTALL_PREFIX must name the isolated SDK install prefix")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${HOLOSCAN_TEST_INSTALL_PREFIX}")
find_package(holoscan
    PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}"
    NO_DEFAULT_PATH
    REQUIRED
)

# Supported Holoscan headers must remain transport-opaque. Scan the isolated install rather than
# the source tree so missing install filters and generated-header leakage are covered too.
file(GLOB_RECURSE _holoscan_installed_public_headers
    LIST_DIRECTORIES FALSE
    "${HOLOSCAN_TEST_INSTALL_PREFIX}/include/holoscan/*.h"
    "${HOLOSCAN_TEST_INSTALL_PREFIX}/include/holoscan/*.hpp"
)
foreach(_holoscan_installed_public_header IN LISTS _holoscan_installed_public_headers)
    file(READ "${_holoscan_installed_public_header}" _holoscan_installed_public_header_text)
    if(_holoscan_installed_public_header_text MATCHES
       "#[ \t]*include[ \t]*[<\"](holomq|iceoryx2|iox2?|iceoryx)/|holomq::|iceoryx2::|iox2?::")
        message(FATAL_ERROR
            "Installed Holoscan header exposes a transport implementation: "
            "${_holoscan_installed_public_header}"
        )
    endif()
    if(_holoscan_installed_public_header_text MATCHES
       "CatalogClockHealthState|CatalogReceivedOperation" OR
       _holoscan_installed_public_header_text MATCHES
       "kHealthControl(Magic|Version|Kind|Bytes)" OR
       _holoscan_installed_public_header_text MATCHES
       "encode_health_control|decode_health_control")
        message(FATAL_ERROR
            "Installed Holoscan header exposes private clock-health record encoding: "
            "${_holoscan_installed_public_header}"
        )
    endif()
endforeach()
unset(_holoscan_installed_public_header_text)
unset(_holoscan_installed_public_header)
unset(_holoscan_installed_public_headers)

# OT-T28 requires stronger evidence for the two public Tensor-loan facades than the general
# transport-namespace scan above. Strip documentation first because the public contract is
# expected to explain which private identities are deliberately absent.
foreach(_holoscan_tensor_loan_header IN ITEMS
        "${HOLOSCAN_TEST_INSTALL_PREFIX}/include/holoscan/core/tensor_output_loan.hpp"
        "${HOLOSCAN_TEST_INSTALL_PREFIX}/include/holoscan/channel/tensor_loan.hpp")
    if(NOT EXISTS "${_holoscan_tensor_loan_header}")
        message(FATAL_ERROR "Installed Tensor-loan header is missing: ${_holoscan_tensor_loan_header}")
    endif()
    file(READ "${_holoscan_tensor_loan_header}" _holoscan_tensor_loan_declarations)
    string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" ""
        _holoscan_tensor_loan_declarations "${_holoscan_tensor_loan_declarations}")
    string(REGEX REPLACE "//[^\r\n]*" ""
        _holoscan_tensor_loan_declarations "${_holoscan_tensor_loan_declarations}")
    foreach(_holoscan_forbidden_tensor_loan_token IN ITEMS
            holomq
            BufferHandle
            PoolId
            SlotId
            EndpointGeneration
            endpoint_generation
            cudaEvent
            CUevent
            IpcMemHandle
            ExternalMemoryHandle
            TransportControl)
        string(REGEX MATCH
            "(^|[^A-Za-z0-9_])${_holoscan_forbidden_tensor_loan_token}([^A-Za-z0-9_]|$)"
            _holoscan_forbidden_tensor_loan_match
            "${_holoscan_tensor_loan_declarations}")
        if(_holoscan_forbidden_tensor_loan_match)
            message(FATAL_ERROR
                "Installed Tensor-loan declaration exposes private token "
                "'${_holoscan_forbidden_tensor_loan_token}': ${_holoscan_tensor_loan_header}")
        endif()
    endforeach()
endforeach()
unset(_holoscan_forbidden_tensor_loan_match)
unset(_holoscan_forbidden_tensor_loan_token)
unset(_holoscan_tensor_loan_declarations)
unset(_holoscan_tensor_loan_header)

if(NOT TARGET holoscan::core)
    message(FATAL_ERROR "Installed package did not define holoscan::core")
endif()

get_target_property(_holoscan_core_imported_type holoscan::core TYPE)
get_target_property(_holoscan_core_interface_links holoscan::core INTERFACE_LINK_LIBRARIES)
if(_holoscan_core_imported_type STREQUAL "SHARED_LIBRARY")
    # HoloMQ is independently linkable from the SDK package. Its imported target may therefore
    # be present after find_package(holoscan), but shared Core must not require consumers to link it.
    if(_holoscan_core_interface_links MATCHES "holomq")
        message(FATAL_ERROR
            "Shared holoscan::core exposes HoloMQ through its link interface"
        )
    endif()
endif()
unset(_holoscan_core_interface_links)
unset(_holoscan_core_imported_type)

add_executable(compile_installed_consumer main.cpp)
target_compile_features(compile_installed_consumer PRIVATE cxx_std_20)
set_target_properties(compile_installed_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(compile_installed_consumer PRIVATE holoscan::core)

include(CTest)
add_test(NAME compile_installed_consumer.run COMMAND compile_installed_consumer)
if(WIN32)
    set_tests_properties(compile_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/bin")
elseif(APPLE)
    set_tests_properties(compile_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "DYLD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
else()
    set_tests_properties(compile_installed_consumer.run PROPERTIES
        ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
endif()
