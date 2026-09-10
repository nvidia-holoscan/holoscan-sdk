# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

function(holoscan_configure_cuda_execution_test_clang_tidy target_name)
    if(HOLOSCAN_ENABLE_CLANG_TIDY AND CMAKE_CXX_CLANG_TIDY)
        # Keep high-signal clang-tidy checks on test targets, but disable checks made noisy by
        # GTest assertion macro expansions.
        set(_holoscan_cuda_execution_test_clang_tidy_checks
            "${CLANG_TIDY_CHECKS},-readability-function-cognitive-complexity"
        )
        set_target_properties(${target_name}
            PROPERTIES
                CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY};--checks=${_holoscan_cuda_execution_test_clang_tidy_checks}"
        )
        unset(_holoscan_cuda_execution_test_clang_tidy_checks)
    endif()
endfunction()

add_executable(cuda_execution_test
    test_cuda_event_pool.cpp
    test_cuda_green_context.cpp
    test_cuda_green_context_driver.cpp
    test_cuda_stream_pool.cpp
    ../common/allocation_counter.cpp
)
holoscan_configure_cuda_execution_test_clang_tidy(cuda_execution_test)

target_link_libraries(cuda_execution_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

target_include_directories(cuda_execution_test
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
)

gtest_discover_tests(cuda_execution_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX CUDA_EXECUTION.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)
