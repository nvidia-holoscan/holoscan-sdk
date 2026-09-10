// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared gtest main implementing the CTest skip protocol (SKIP_RETURN_CODE 77 in
// tests/unit/CMakeLists.txt). gtest exits 0 when tests are skipped, and a CTest
// skip-output regex would override a real failure elsewhere in the same binary, so the
// classification lives here: failures always win (exit 1), a binary whose every selected
// test skipped reports 77 (CTest: Skipped), and a partially-skipped binary reports 0
// (CTest: Passed).

#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  const ::testing::UnitTest& unit = *::testing::UnitTest::GetInstance();
  if (rc != 0 || unit.failed_test_count() > 0) {
    return 1;
  }
  const int ran = unit.test_to_run_count();
  if (ran > 0 && unit.skipped_test_count() == ran) {
    return 77;
  }
  return 0;
}
