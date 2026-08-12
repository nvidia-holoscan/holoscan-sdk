# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

if(TARGET params)
    add_executable(params_test
        test_config_binding.cpp
        test_params.cpp
    )
    holoscan_configure_test_clang_tidy(params_test)
    holoscan_apply_logging_policy(params_test)
    target_include_directories(params_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(params_test
        PRIVATE
            holoscan::params
            GTest::gtest_main
    )
    if(TARGET config_io)
        target_sources(params_test PRIVATE test_config_binding_provenance.cpp)
        target_link_libraries(params_test PRIVATE holoscan::config_io)
    endif()
    gtest_discover_tests(params_test
        DISCOVERY_MODE PRE_TEST
        TEST_PREFIX PARAMS.
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

if(TARGET params_io)
    add_executable(params_io_test test_params_io.cpp)
    holoscan_configure_test_clang_tidy(params_io_test)
    holoscan_apply_logging_policy(params_io_test)
    target_include_directories(params_io_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(params_io_test
        PRIVATE
            holoscan::params_io
            GTest::gtest_main
    )
    gtest_discover_tests(params_io_test
        DISCOVERY_MODE PRE_TEST
        TEST_PREFIX PARAMS_IO.
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

if(TARGET experimental_params_web_tuner)
    add_executable(web_tuner_test test_web_tuner.cpp)
    holoscan_configure_test_clang_tidy(web_tuner_test)
    holoscan_apply_logging_policy(web_tuner_test)
    target_include_directories(web_tuner_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(web_tuner_test
        PRIVATE
            holoscan::experimental_params_web_tuner
            GTest::gtest_main
    )
    gtest_discover_tests(web_tuner_test
        DISCOVERY_MODE PRE_TEST
        TEST_PREFIX WEB_TUNER.
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
