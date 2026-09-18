# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_descriptor_tensor_external_topic LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

holoscan_add_flatbuffer_schema(
    TARGET descriptor_tensor_external_topic_generated
    SCHEMA external_frame.fbs
    MAX_SERIALIZED_SIZE 512
)

set(DESCRIPTOR_TOPIC_TARGETS
    descriptor_tensor_external_topic_graph
    descriptor_tensor_external_topic_client
)
foreach(target IN LISTS DESCRIPTOR_TOPIC_TARGETS)
    add_executable(${target} "${target}.cpp")
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_link_libraries(${target}
        PRIVATE holoscan::core descriptor_tensor_external_topic_generated)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    if(COMMAND holoscan_configure_example_install)
        holoscan_configure_example_install(${target})
    endif()
endforeach()

if(BUILD_TESTING)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME EXAMPLE_CPP_DESCRIPTOR_TENSOR_EXTERNAL_TOPIC_TEST
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_example.py"
                    --graph $<TARGET_FILE:descriptor_tensor_external_topic_graph>
                    --client $<TARGET_FILE:descriptor_tensor_external_topic_client>
        )
        set_tests_properties(EXAMPLE_CPP_DESCRIPTOR_TENSOR_EXTERNAL_TOPIC_TEST
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "descriptor Tensor external-topic round trip complete"
                LABELS "examples;schema;tensor;multi-tensor;external-topic;interprocess"
                TIMEOUT 60
        )
        add_test(
            NAME EXAMPLE_CPP_DESCRIPTOR_TENSOR_EXTERNAL_TOPIC_RUNNER_TEST
            COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                    "${CMAKE_CURRENT_SOURCE_DIR}/test_run_example.py"
        )
        set_tests_properties(EXAMPLE_CPP_DESCRIPTOR_TENSOR_EXTERNAL_TOPIC_RUNNER_TEST
            PROPERTIES
                LABELS "examples;schema;tensor;external-topic;python"
                TIMEOUT 30
        )
    endif()
endif()

install(TARGETS ${DESCRIPTOR_TOPIC_TARGETS}
    RUNTIME DESTINATION
        "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/flatbuffer_schemas/descriptor_tensor_external_topic/cpp"
    COMPONENT holoscan-core-examples
)
# The examples source install matches only C++, CMake, schema, and document extensions, so every
# Python file this directory's tests read must be installed explicitly. test_run_example.py imports
# run_example as a sibling module, so both stay in the same installed directory.
install(FILES external_frame.fbs external_topic_common.hpp test_run_example.py
    DESTINATION
        "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/flatbuffer_schemas/descriptor_tensor_external_topic/cpp"
    COMPONENT holoscan-core-examples
)
install(PROGRAMS run_example.py
    DESTINATION
        "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/flatbuffer_schemas/descriptor_tensor_external_topic/cpp"
    COMPONENT holoscan-core-examples
)
