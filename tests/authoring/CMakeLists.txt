# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

add_executable(authoring_registry_test
    ../common/allocation_counter.cpp
    test_parameter_materializer.cpp
    test_payload_clock_registry.cpp
    test_registry.cpp
    test_thunk_authoring.cpp
)
holoscan_configure_test_clang_tidy(authoring_registry_test)

target_link_libraries(authoring_registry_test
    PRIVATE
        holoscan::core
        holoscan::schema
        GTest::gtest_main
)
if(HOLOSCAN_BUILD_PARAMS)
    target_link_libraries(authoring_registry_test PRIVATE holoscan::params)
    target_compile_definitions(authoring_registry_test PRIVATE HOLOSCAN_AUTHORING_TEST_PARAMS=1)
endif()

gtest_discover_tests(authoring_registry_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX AUTHORING_REGISTRY.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

function(holoscan_add_authoring_compile_fail_test test_name source_name diagnostic_fragment)
    set(target_name "authoring_compile_fail_${test_name}")
    set(source_path "compile_fail/${source_name}")
    add_executable(${target_name} EXCLUDE_FROM_ALL "${source_path}")
    set_source_files_properties("${source_path}" PROPERTIES SKIP_LINTING ON)
    target_link_libraries(${target_name} PRIVATE holoscan::core)
    add_test(NAME "AUTHORING_COMPILE_FAIL.${test_name}"
        COMMAND ${CMAKE_COMMAND}
            -DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}
            -DCMAKE_BUILD_TYPE=$<CONFIG>
            -DHOLOSCAN_TARGET=${target_name}
            -DHOLOSCAN_EXPECTED_DIAGNOSTIC=${diagnostic_fragment}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/../core/expected_compile_fail/verify_compile_failure.cmake
    )
    set_tests_properties("AUTHORING_COMPILE_FAIL.${test_name}" PROPERTIES
        RESOURCE_LOCK authoring_compile_fail_build
    )
endfunction()

holoscan_add_authoring_compile_fail_test(
    opaque_params
    opaque_params.cpp
    "HOLOSCAN_REGISTER_OPERATOR requires a declared schema for configurable Params"
)
holoscan_add_authoring_compile_fail_test(
    non_default_operator
    non_default_operator.cpp
    "HOLOSCAN_REGISTER_OPERATOR requires a default-constructible Operator"
)

holoscan_add_authoring_compile_fail_test(
    local_enum_payload_factory
    local_enum_payload_factory.cpp
    "ExternalTopicPayload"
)

holoscan_add_authoring_compile_fail_test(
    local_enum_payload_automatic
    local_enum_payload_automatic.cpp
    "ExternalTopicPayload"
)

holoscan_add_authoring_compile_fail_test(
    local_enum_payload_macro
    local_enum_payload_macro.cpp
    "ExternalTopicPayload"
)

holoscan_add_authoring_compile_fail_test(
    local_enum_payload_thunks
    local_enum_payload_thunks.cpp
    "ExternalTopicPayload"
)
