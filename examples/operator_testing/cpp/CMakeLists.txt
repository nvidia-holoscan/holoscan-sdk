# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_operator_testing LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(operator_testing operator_testing.cpp)
target_link_libraries(operator_testing PRIVATE holoscan::testing)
holoscan_configure_example_install(operator_testing)
if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_OPERATOR_TESTING_TEST COMMAND operator_testing)
    set_tests_properties(EXAMPLE_CPP_OPERATOR_TESTING_TEST PROPERTIES
        LABELS "examples;testing" TIMEOUT 60
        PASS_REGULAR_EXPRESSION "operator_testing example complete")
endif()
install(TARGETS operator_testing
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/operator_testing/cpp"
    COMPONENT holoscan-testing-examples)

add_executable(operator_testing_tensor tensor_snapshot.cpp)
target_link_libraries(operator_testing_tensor PRIVATE holoscan::testing)
holoscan_configure_example_install(operator_testing_tensor)
add_executable(operator_testing_image image_snapshot.cpp)
target_link_libraries(operator_testing_image PRIVATE holoscan::testing holoscan::sensor_io)
holoscan_configure_example_install(operator_testing_image)
if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_OPERATOR_TESTING_TENSOR_TEST COMMAND operator_testing_tensor)
    add_test(NAME EXAMPLE_CPP_OPERATOR_TESTING_IMAGE_TEST COMMAND operator_testing_image)
    set_tests_properties(EXAMPLE_CPP_OPERATOR_TESTING_TENSOR_TEST EXAMPLE_CPP_OPERATOR_TESTING_IMAGE_TEST
        PROPERTIES LABELS "examples;testing" TIMEOUT 60)
endif()
install(TARGETS operator_testing_tensor operator_testing_image
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/operator_testing/cpp"
    COMPONENT holoscan-testing-examples)

set(OPERATOR_TESTING_DEVICE_ID "0" CACHE STRING "CUDA device ordinal used by operator testing examples")
add_executable(operator_testing_device device_inputs.cpp)
target_link_libraries(operator_testing_device PRIVATE holoscan::testing)
if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_OPERATOR_TESTING_DEVICE_TEST
        COMMAND operator_testing_device "${OPERATOR_TESTING_DEVICE_ID}")
    add_test(NAME EXAMPLE_CPP_OPERATOR_TESTING_GREEN_CONTEXT_TEST
        COMMAND operator_testing_device "${OPERATOR_TESTING_DEVICE_ID}" --green-context)
    set_tests_properties(EXAMPLE_CPP_OPERATOR_TESTING_DEVICE_TEST
        EXAMPLE_CPP_OPERATOR_TESTING_GREEN_CONTEXT_TEST
        PROPERTIES LABELS "examples;testing;cuda" TIMEOUT 60 SKIP_RETURN_CODE 77)
endif()
holoscan_configure_example_install(operator_testing_device)
install(TARGETS operator_testing_device
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/operator_testing/cpp"
    COMPONENT holoscan-testing-examples)


holoscan_add_flatbuffer_schema(
    TARGET operator_testing_schema_generated
    SCHEMA paired_frame.fbs
    MAX_SERIALIZED_SIZE 768)
add_executable(operator_testing_schema schema_snapshot.cpp)
target_link_libraries(operator_testing_schema PRIVATE holoscan::testing operator_testing_schema_generated)
add_executable(operator_testing_lifecycle lifecycle_outputs.cpp)
target_link_libraries(operator_testing_lifecycle PRIVATE holoscan::testing)
if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_OPERATOR_TESTING_SCHEMA_TEST COMMAND operator_testing_schema)
    add_test(NAME EXAMPLE_CPP_OPERATOR_TESTING_LIFECYCLE_TEST COMMAND operator_testing_lifecycle)
    set_tests_properties(EXAMPLE_CPP_OPERATOR_TESTING_SCHEMA_TEST EXAMPLE_CPP_OPERATOR_TESTING_LIFECYCLE_TEST
        PROPERTIES LABELS "examples;testing" TIMEOUT 60)
endif()
holoscan_configure_example_install(operator_testing_schema)
holoscan_configure_example_install(operator_testing_lifecycle)
install(TARGETS operator_testing_schema operator_testing_lifecycle
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/operator_testing/cpp"
    COMPONENT holoscan-testing-examples)

add_executable(operator_testing_clock controlled_clock.cpp)
target_link_libraries(operator_testing_clock PRIVATE holoscan::testing)
if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_OPERATOR_TESTING_CLOCK_TEST COMMAND operator_testing_clock)
    set_tests_properties(EXAMPLE_CPP_OPERATOR_TESTING_CLOCK_TEST
        PROPERTIES LABELS "examples;testing" TIMEOUT 60)
endif()
holoscan_configure_example_install(operator_testing_clock)
install(TARGETS operator_testing_clock
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/operator_testing/cpp"
    COMPONENT holoscan-testing-examples)

add_executable(operator_testing_restart managed_restart.cpp)
target_link_libraries(operator_testing_restart PRIVATE holoscan::testing)
if(BUILD_TESTING)
    add_test(NAME EXAMPLE_CPP_OPERATOR_TESTING_RESTART_TEST COMMAND operator_testing_restart)
    set_tests_properties(EXAMPLE_CPP_OPERATOR_TESTING_RESTART_TEST
        PROPERTIES LABELS "examples;testing" TIMEOUT 60)
endif()
holoscan_configure_example_install(operator_testing_restart)
install(TARGETS operator_testing_restart
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/operator_testing/cpp"
    COMPONENT holoscan-testing-examples)
