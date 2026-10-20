# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
project(operator_testing_installed_consumer LANGUAGES CXX)
find_package(holoscan COMPONENTS testing schema sensor_io REQUIRED)
include(GenerateExportHeader)
# A static SDK must have one linked Core instance. Loading a second static Core in a
# shared provider duplicates Tensor Runtime type tables; qualify DSO isolation with shared Core.
get_target_property(core_library_type holoscan::core TYPE)
if(core_library_type STREQUAL "STATIC_LIBRARY")
    add_library(test_operator_provider STATIC provider.cpp)
    target_compile_definitions(test_operator_provider PUBLIC
        TEST_OPERATOR_PROVIDER_STATIC_DEFINE TEST_OPERATOR_PROVIDER_SHARED=0)
else()
    add_library(test_operator_provider SHARED provider.cpp)
    target_compile_definitions(test_operator_provider PUBLIC TEST_OPERATOR_PROVIDER_SHARED=1)
endif()
generate_export_header(test_operator_provider)
target_compile_features(test_operator_provider PUBLIC cxx_std_20)
target_include_directories(test_operator_provider PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(test_operator_provider PUBLIC
    holoscan::testing holoscan::schema holoscan::sensor_io)
set_target_properties(test_operator_provider PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
add_executable(operator_testing_consumer main.cpp)
set_target_properties(operator_testing_consumer PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
target_link_libraries(operator_testing_consumer PRIVATE holoscan::testing test_operator_provider)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/consumer-path-$<CONFIG>.txt"
    CONTENT "$<TARGET_FILE:operator_testing_consumer>")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/provider-mode-$<CONFIG>.txt"
    CONTENT "$<TARGET_PROPERTY:test_operator_provider,TYPE>\n")
