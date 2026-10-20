# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

function(holoscan_configure_memory_test_clang_tidy target_name)
    if(HOLOSCAN_ENABLE_CLANG_TIDY AND CMAKE_CXX_CLANG_TIDY)
        # Keep high-signal clang-tidy checks on test targets, but disable checks made noisy by
        # GTest assertion macro expansions.
        set(_holoscan_memory_test_clang_tidy_checks
            "${CLANG_TIDY_CHECKS},-readability-function-cognitive-complexity"
        )
        set_target_properties(${target_name}
            PROPERTIES
                CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY};--checks=${_holoscan_memory_test_clang_tidy_checks}"
        )
        unset(_holoscan_memory_test_clang_tidy_checks)
    endif()
endfunction()

add_executable(memory_test
    test_block_memory_pool.cpp
    test_cuda_memory_resource_allocator.cpp
    test_memory_allocator.cpp
    test_memory_buffer.cpp
    test_unbounded_allocator.cpp
    ../common/allocation_counter.cpp
)
holoscan_configure_memory_test_clang_tidy(memory_test)
target_include_directories(memory_test PRIVATE ${holoscan_SOURCE_DIR}/src)

target_link_libraries(memory_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(memory_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX MEMORY.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)
