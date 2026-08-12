/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

holoscan::expected<int, holoscan::Error> value_result() {
  return 42;
}

// REQ: EXPECTED-019
holoscan::expected<int, holoscan::Error> invalid_unbraced_value_control_flow(const bool run) {
  if (run)
    HOLOSCAN_TRY(auto value, value_result());
  return 0;
}

int main() {
  (void)invalid_unbraced_value_control_flow(true);
  return 0;
}
