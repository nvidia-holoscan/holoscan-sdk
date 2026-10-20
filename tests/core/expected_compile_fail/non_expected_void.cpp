/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/expected.hpp>

// REQ: EXPECTED-009
int invalid_void_operand() {
  HOLOSCAN_TRY(42);
  return 0;
}

int main() {
  return invalid_void_operand();
}
