# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)
project(holoscan_installed_byte_provider LANGUAGES CXX)

find_package(holoscan REQUIRED CONFIG PATHS "${HOLOSCAN_TEST_INSTALL_PREFIX}" NO_DEFAULT_PATH)
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development.SABIModule)
holoscan_add_flatbuffer_schema(
    TARGET external_byte_schema
    SCHEMA byte_channel_payload.fbs
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated"
    MAX_SERIALIZED_SIZE 1024
    PYTHON
)
add_executable(byte_schema_peer peer.cpp)
target_compile_features(byte_schema_peer PRIVATE cxx_std_20)
target_link_libraries(byte_schema_peer PRIVATE holoscan::core external_byte_schema)
set_target_properties(byte_schema_peer PROPERTIES CXX_EXTENSIONS OFF)

enable_testing()
add_test(NAME installed_byte_provider
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/check_provider.py"
            $<TARGET_FILE:byte_schema_peer> "${HOLOSCAN_TEST_INSTALL_PREFIX}")
set(site_packages "${HOLOSCAN_TEST_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
set_tests_properties(installed_byte_provider PROPERTIES TIMEOUT 30
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/generated:${site_packages};LD_LIBRARY_PATH=${HOLOSCAN_TEST_INSTALL_PREFIX}/lib")
