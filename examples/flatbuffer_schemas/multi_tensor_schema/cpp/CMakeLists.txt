# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_multi_tensor_schema LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

holoscan_add_flatbuffer_schema(
    TARGET multi_tensor_schema_generated
    SCHEMA stereo_frame.fbs
    MAX_SERIALIZED_SIZE 768
    NATIVE_BOUNDS_FILE native_bounds.json
)

add_executable(multi_tensor_schema multi_tensor_schema.cpp)
target_compile_features(multi_tensor_schema PRIVATE cxx_std_20)
target_link_libraries(multi_tensor_schema
    PRIVATE CUDA::cudart holoscan::core multi_tensor_schema_generated)
set_target_properties(multi_tensor_schema PROPERTIES CXX_EXTENSIONS OFF)

if(COMMAND holoscan_configure_example_install)
    holoscan_configure_example_install(multi_tensor_schema)
endif()

if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_MULTI_TENSOR_SCHEMA_TEST COMMAND multi_tensor_schema)
    set_tests_properties(EXAMPLE_CPP_MULTI_TENSOR_SCHEMA_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "multi_tensor_schema: transformed one prepared StereoFrameT sample"
            SKIP_RETURN_CODE 77
            LABELS "examples;schema;tensor;multi-tensor;cuda"
            TIMEOUT 30)
endif()

install(TARGETS multi_tensor_schema
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/flatbuffer_schemas/multi_tensor_schema/cpp"
    COMPONENT holoscan-core-examples
)
install(FILES stereo_frame.fbs native_bounds.json
    DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/flatbuffer_schemas/multi_tensor_schema/cpp"
    COMPONENT holoscan-core-examples
)
