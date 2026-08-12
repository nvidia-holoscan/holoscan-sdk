/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/predicate.hpp>

// REQ: CONTRACT-010
auto invalid_predicate = holoscan::predicate("invalid", []() { return 42; });

int main() {
  return 0;
}
