# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET GTest::gtest_main)
    find_package(GTest CONFIG QUIET)
endif()
if(NOT TARGET GTest::gtest_main)
    if(PROJECT_IS_TOP_LEVEL)
        set(_holomq_gtest_component "${HOLOMQ_INSTALL_COMPONENT}")
    else()
        set(_holomq_gtest_component "holoscan-dependencies")
    endif()
    holoscan_add_dependency(gtest_rapids COMPONENT ${_holomq_gtest_component})
    unset(_holomq_gtest_component)
endif()

if(TARGET holomq::holomq_internals)
    set(HOLOMQ_TEST_LIBRARY holomq::holomq_internals)
else()
    set(HOLOMQ_TEST_LIBRARY holomq::holomq)
endif()

add_subdirectory(unit)
