# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

function(holoscan_configure_threading_test_clang_tidy target_name)
    if(HOLOSCAN_ENABLE_CLANG_TIDY AND CMAKE_CXX_CLANG_TIDY)
        # Keep high-signal clang-tidy checks on test targets, but disable checks made noisy by
        # GTest assertion macro expansions.
        set(_holoscan_threading_test_clang_tidy_checks
            "${CLANG_TIDY_CHECKS},-readability-function-cognitive-complexity"
        )
        set_target_properties(${target_name}
            PROPERTIES
                CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY};--checks=${_holoscan_threading_test_clang_tidy_checks}"
        )
        unset(_holoscan_threading_test_clang_tidy_checks)
    endif()
endfunction()

add_executable(threading_test
    ${CMAKE_SOURCE_DIR}/src/threading/linux_thread_controller.cpp
    test_linux_thread_controller.cpp
    test_thread_placement.cpp
)
holoscan_configure_threading_test_clang_tidy(threading_test)

target_include_directories(threading_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(threading_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(threading_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX THREADING.
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
