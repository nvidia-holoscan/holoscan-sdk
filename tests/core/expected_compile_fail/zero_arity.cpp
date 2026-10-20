/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

holoscan::expected<void, holoscan::Error> invalid_zero_arity() {
  // REQ: EXPECTED-012
  HOLOSCAN_TRY();
  return {};
}
