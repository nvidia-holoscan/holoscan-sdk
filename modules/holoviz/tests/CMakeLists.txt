# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT TARGET GTest::gtest_main)
   find_package(GTest)
endif()

if(TARGET GTest::gtest_main OR GTest_FOUND)
   add_subdirectory(functional)
   add_subdirectory(unit)
endif()
