/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

// REQ: EXPECTED-011
holoscan::expected<int, holoscan::Error> invalid_value_form() {
  HOLOSCAN_ASSIGN_OR_RETURN(auto value, (holoscan::expected<void, holoscan::Error>{}));
  return value;
}
