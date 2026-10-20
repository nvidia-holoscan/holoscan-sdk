/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

holoscan::expected<int, holoscan::Error> value_result() {
  return 1;
}

// REQ: EXPECTED-012
holoscan::expected<void, holoscan::Error> invalid_arity() {
  HOLOSCAN_TRY(auto first, value_result(), auto second);
  return {};
}
