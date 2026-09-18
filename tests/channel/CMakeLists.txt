# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

function(holoscan_configure_channel_test_clang_tidy target_name)
    if(HOLOSCAN_ENABLE_CLANG_TIDY AND CMAKE_CXX_CLANG_TIDY)
        # Keep high-signal clang-tidy checks on test targets, but disable checks made noisy by
        # GTest assertion macro expansions.
        set(_holoscan_channel_test_clang_tidy_checks
            "${CLANG_TIDY_CHECKS},-readability-function-cognitive-complexity"
        )
        set_target_properties(${target_name}
            PROPERTIES
                CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY};--checks=${_holoscan_channel_test_clang_tidy_checks}"
        )
        unset(_holoscan_channel_test_clang_tidy_checks)
    endif()
endfunction()

function(holoscan_add_channel_expected_compile_fail_test
         test_name source_name diagnostic_fragment)
    set(target_name "channel_expected_compile_fail_${test_name}")
    set(source_path "expected_compile_fail/${source_name}")
    add_executable(${target_name} EXCLUDE_FROM_ALL "${source_path}")
    set_source_files_properties("${source_path}" PROPERTIES SKIP_LINTING ON)
    target_link_libraries(${target_name} PRIVATE holoscan::core)

    add_test(NAME "CHANNEL_EXPECTED_COMPILE_FAIL.${test_name}"
        COMMAND ${CMAKE_COMMAND}
            -DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}
            -DCMAKE_BUILD_TYPE=$<CONFIG>
            -DHOLOSCAN_TARGET=${target_name}
            -DHOLOSCAN_EXPECTED_DIAGNOSTIC=${diagnostic_fragment}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/../core/expected_compile_fail/verify_compile_failure.cmake
    )
    # Building a negative fixture may relink libraries used by running SDK tests.
    set_tests_properties("CHANNEL_EXPECTED_COMPILE_FAIL.${test_name}" PROPERTIES RUN_SERIAL TRUE)
endfunction()

add_library(channel_process_test_support STATIC
    channel_test_support.cpp
)
holoscan_configure_channel_test_clang_tidy(channel_process_test_support)
target_link_libraries(channel_process_test_support PRIVATE holoscan::core)

add_library(channel_holomq_test_support STATIC
    channel_holomq_test_support.cpp
)
holoscan_configure_channel_test_clang_tidy(channel_holomq_test_support)
target_link_libraries(channel_holomq_test_support
    PRIVATE
        GTest::gtest
        holoscan::core
        holomq::holomq
)

holoscan_add_channel_expected_compile_fail_test(
    runtime_options_removed
    runtime_options_removed.cpp
    "ChannelRuntimeOptions"
)
holoscan_add_channel_expected_compile_fail_test(
    transport_capabilities_removed
    transport_capabilities_removed.cpp
    "transport_capabilities"
)

add_executable(channel_test
    test_catalog_channel.cpp
    test_catalog_pod_interprocess.cpp
    test_channel.cpp
    test_channel_tensor_loan.cpp
    test_channel_wait.cpp
    test_pod_codec.cpp
    ../common/allocation_counter.cpp
    ../../src/channel/channel_common.cpp
)
holoscan_configure_channel_test_clang_tidy(channel_test)
target_include_directories(channel_test PRIVATE ${holoscan_SOURCE_DIR}/src)

add_executable(channel_interprocess_child
    channel_interprocess_child.cpp
)
holoscan_configure_channel_test_clang_tidy(channel_interprocess_child)

add_executable(channel_catalog_pod_interprocess_child
    catalog_pod_interprocess_child.cpp
)
holoscan_configure_channel_test_clang_tidy(channel_catalog_pod_interprocess_child)

add_executable(channel_event_wait_child
    channel_event_wait_child.cpp
)
holoscan_configure_channel_test_clang_tidy(channel_event_wait_child)

target_link_libraries(channel_test
    PRIVATE
        channel_process_test_support
        flatbuffers::flatbuffers
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        holomq::holomq
        GTest::gtest_main
)
add_dependencies(channel_test
    holoscan_flatbuffer_schemas_generated
)
target_link_libraries(channel_interprocess_child
    PRIVATE
        channel_process_test_support
        holoscan::core
        holomq::holomq
)
target_link_libraries(channel_catalog_pod_interprocess_child
    PRIVATE
        channel_process_test_support
        holoscan::core
        holomq::holomq
)
target_link_libraries(channel_event_wait_child
    PRIVATE
        channel_process_test_support
        holoscan::core
        holomq::holomq
)
target_compile_definitions(channel_test
    PRIVATE
        HOLOSCAN_CHANNEL_INTERPROCESS_CHILD_EXE="$<TARGET_FILE:channel_interprocess_child>"
        HOLOSCAN_CHANNEL_CATALOG_POD_INTERPROCESS_CHILD_EXE="$<TARGET_FILE:channel_catalog_pod_interprocess_child>"
        HOLOSCAN_CHANNEL_EVENT_WAIT_CHILD_EXE="$<TARGET_FILE:channel_event_wait_child>"
)
add_dependencies(channel_test
    channel_catalog_pod_interprocess_child
    channel_event_wait_child
    channel_interprocess_child
)

gtest_discover_tests(channel_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX CHANNEL.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(channel_schema_test
    test_schema.cpp
    test_schema_descriptor_tensor.cpp
    test_schema_native.cpp
    test_schema_package_validation.cpp
    ../common/allocation_counter.cpp
)
holoscan_configure_channel_test_clang_tidy(channel_schema_test)

set(HOLOSCAN_CHANNEL_TEST_SCHEMA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
holoscan_add_flatbuffer_schema(
    TARGET channel_test_flatbuffer_schemas_generated
    SCHEMAS custom_control.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${HOLOSCAN_CHANNEL_TEST_SCHEMA_GENERATED_DIR}"
    TRAITS_HEADER custom_control_schema_traits.hpp
    MAX_SERIALIZED_SIZE 512
)
add_dependencies(channel_schema_test
    holoscan_flatbuffer_schemas_generated
    channel_test_flatbuffer_schemas_generated
)

add_executable(channel_schema_interprocess_child
    schema_interprocess_child.cpp
)
holoscan_configure_channel_test_clang_tidy(channel_schema_interprocess_child)
add_dependencies(channel_schema_interprocess_child
    holoscan_flatbuffer_schemas_generated
    channel_test_flatbuffer_schemas_generated
)

target_include_directories(channel_schema_test
    PRIVATE
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
        "${HOLOSCAN_CHANNEL_TEST_SCHEMA_GENERATED_DIR}"
)
target_include_directories(channel_schema_interprocess_child
    PRIVATE
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
        "${HOLOSCAN_CHANNEL_TEST_SCHEMA_GENERATED_DIR}"
)

target_link_libraries(channel_schema_test
    PRIVATE
        channel_holomq_test_support
        channel_process_test_support
        flatbuffers::flatbuffers
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        holomq::holomq
        GTest::gtest_main
)
target_link_libraries(channel_schema_interprocess_child
    PRIVATE
        channel_process_test_support
        flatbuffers::flatbuffers
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        holomq::holomq
)
target_compile_definitions(channel_schema_test
    PRIVATE
        HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE="$<TARGET_FILE:channel_schema_interprocess_child>"
)
add_dependencies(channel_schema_test
    channel_schema_interprocess_child
)

gtest_discover_tests(channel_schema_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX CHANNEL_SCHEMA.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(channel_tensor_test
    test_channel_tensor.cpp
    ../common/allocation_counter.cpp
    ../../src/channel/channel_common.cpp
)
holoscan_configure_channel_test_clang_tidy(channel_tensor_test)

add_executable(channel_tensor_interprocess_child
    channel_tensor_interprocess_child.cpp
)
holoscan_configure_channel_test_clang_tidy(channel_tensor_interprocess_child)

target_link_libraries(channel_tensor_test
    PRIVATE
        channel_holomq_test_support
        channel_process_test_support
        holoscan::core
        holomq::holomq
        GTest::gtest_main
)
target_link_libraries(channel_tensor_interprocess_child
    PRIVATE
        channel_process_test_support
        holoscan::core
        holomq::holomq
)
target_compile_definitions(channel_tensor_test
    PRIVATE
        HOLOSCAN_CHANNEL_TENSOR_INTERPROCESS_CHILD_EXE="$<TARGET_FILE:channel_tensor_interprocess_child>"
)
target_compile_definitions(channel_test
    PRIVATE
        HOLOSCAN_CHANNEL_TENSOR_INTERPROCESS_CHILD_EXE="$<TARGET_FILE:channel_tensor_interprocess_child>"
)
add_dependencies(channel_tensor_test
    channel_tensor_interprocess_child
)
add_dependencies(channel_test
    channel_tensor_interprocess_child
)

gtest_discover_tests(channel_tensor_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX CHANNEL.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)
