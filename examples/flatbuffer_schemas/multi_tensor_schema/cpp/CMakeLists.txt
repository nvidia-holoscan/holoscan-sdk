# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_multi_tensor_schema LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

set(MULTI_TENSOR_SCHEMA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
holoscan_add_flatbuffer_schema(
    TARGET multi_tensor_schema_generated
    SCHEMAS stereo_frame.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${MULTI_TENSOR_SCHEMA_GENERATED_DIR}"
    BFBS_EMBED
    DESCRIPTOR_TENSOR
    TRAITS_HEADER stereo_frame_traits.hpp
    TRAITS_ROOT_TYPE stereo::StereoFrame
    TRAITS_NATIVE_TYPE stereo::StereoFrameT
    TRAITS_BINARY_SCHEMA_TYPE stereo::StereoFrameBinarySchema
    TRAITS_PACKAGE_NAMESPACE stereo::package
    TRAITS_CANONICAL_NAME stereo.StereoFrame
    TRAITS_SCHEMA_VERSION 1
    TRAITS_MAX_SERIALIZED_SIZE 1536
    TRAITS_DESCRIPTOR_METADATA_CAPACITY 768
)

add_executable(multi_tensor_schema multi_tensor_schema.cpp)
add_dependencies(multi_tensor_schema multi_tensor_schema_generated)
target_compile_features(multi_tensor_schema PRIVATE cxx_std_20)
target_include_directories(multi_tensor_schema
    PRIVATE "${MULTI_TENSOR_SCHEMA_GENERATED_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(multi_tensor_schema
    PRIVATE CUDA::cudart holoscan::core holoscan::schema)
set_target_properties(multi_tensor_schema PROPERTIES CXX_EXTENSIONS OFF)

if(COMMAND holoscan_configure_example_install)
    holoscan_configure_example_install(multi_tensor_schema)
endif()

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_MULTI_TENSOR_SCHEMA_TEST COMMAND multi_tensor_schema)
    set_tests_properties(EXAMPLE_CPP_MULTI_TENSOR_SCHEMA_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "multi_tensor_schema: received one StereoFrameT sample"
            SKIP_RETURN_CODE 77
            LABELS "examples;schema;tensor;multi-tensor;cuda"
            TIMEOUT 30)
endif()

install(TARGETS multi_tensor_schema
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/flatbuffer_schemas/multi_tensor_schema/cpp"
    COMPONENT holoscan-examples
)
install(FILES stereo_frame.fbs
    DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/flatbuffer_schemas/multi_tensor_schema/cpp"
    COMPONENT holoscan-examples
)
