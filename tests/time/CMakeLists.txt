# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

add_executable(time_test
    clock.cpp
)
holoscan_configure_test_clang_tidy(time_test)

target_link_libraries(time_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(time_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX TIME.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

add_executable(clock_registry_test
    test_clock_registry.cpp
)
holoscan_configure_test_clang_tidy(clock_registry_test)

target_include_directories(clock_registry_test PRIVATE ${holoscan_SOURCE_DIR}/src)

target_link_libraries(clock_registry_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(clock_registry_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX CLOCK_REGISTRY.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)
