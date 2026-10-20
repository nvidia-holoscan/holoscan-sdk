/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

template <typename Left, typename Right>
holoscan::expected<int, holoscan::Error> templated_result() {
  return 42;
}

// REQ: EXPECTED-012
holoscan::expected<int, holoscan::Error> invalid_unprotected_rhs_comma() {
  HOLOSCAN_TRY(auto value, templated_result<int, int>());
  return value;
}
