# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET GTest::gtest_main)
    find_package(GTest CONFIG QUIET)
endif()
if(NOT TARGET GTest::gtest_main)
    holoscan_add_dependency(gtest_rapids COMPONENT holoscan-dependencies)
endif()

# White-box tests consume the library's already-compiled object target directly.
# This keeps private implementation coverage without producing a second HoloMQ
# archive or exposing an internal target to SDK consumers.
set(HOLOMQ_TEST_LIBRARY holomq_objects)

add_subdirectory(unit)
