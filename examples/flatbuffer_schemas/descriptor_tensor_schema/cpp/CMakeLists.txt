# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_descriptor_tensor_schema LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

set(DESCRIPTOR_SCHEMA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
holoscan_add_flatbuffer_schema(
    TARGET descriptor_tensor_schema_generated
    SCHEMAS descriptor_frame.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${DESCRIPTOR_SCHEMA_GENERATED_DIR}"
    BFBS_EMBED
    DESCRIPTOR_TENSOR
    TRAITS_HEADER descriptor_frame_traits.hpp
    TRAITS_ROOT_TYPE app3p::DescriptorFrame
    TRAITS_NATIVE_TYPE app3p::DescriptorFrameT
    TRAITS_BINARY_SCHEMA_TYPE app3p::DescriptorFrameBinarySchema
    TRAITS_PACKAGE_NAMESPACE app3p::package
    TRAITS_CANONICAL_NAME app3p.DescriptorFrame
    TRAITS_SCHEMA_VERSION 1
    TRAITS_MAX_SERIALIZED_SIZE 1024
    TRAITS_DESCRIPTOR_METADATA_CAPACITY 512
)
add_executable(descriptor_tensor_schema descriptor_tensor_schema.cpp)
add_dependencies(descriptor_tensor_schema descriptor_tensor_schema_generated)
target_compile_features(descriptor_tensor_schema PRIVATE cxx_std_20)
target_include_directories(descriptor_tensor_schema
    PRIVATE "${DESCRIPTOR_SCHEMA_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(descriptor_tensor_schema PRIVATE holoscan::core holoscan::schema)
set_target_properties(descriptor_tensor_schema PROPERTIES CXX_EXTENSIONS OFF)

if(COMMAND holoscan_configure_example_install)
    holoscan_configure_example_install(descriptor_tensor_schema)
endif()

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_DESCRIPTOR_TENSOR_SCHEMA_TEST COMMAND descriptor_tensor_schema)
    set_tests_properties(EXAMPLE_CPP_DESCRIPTOR_TENSOR_SCHEMA_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "descriptor_tensor_schema: received one DescriptorFrameT sample"
            LABELS "examples;schema;tensor;third-party"
            TIMEOUT 30)
endif()

install(TARGETS descriptor_tensor_schema
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/flatbuffer_schemas/descriptor_tensor_schema/cpp"
    COMPONENT holoscan-examples
)
install(FILES descriptor_frame.fbs
    DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/flatbuffer_schemas/descriptor_tensor_schema/cpp"
    COMPONENT holoscan-examples
)
