/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

holoscan::expected<int, holoscan::Error> result() {
  return 42;
}

// REQ: EXPECTED-013
int invalid_expression_form() {
  auto value = HOLOSCAN_TRYX(result());
  return value;
}
