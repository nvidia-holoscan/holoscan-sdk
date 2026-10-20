/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

// REQ: EXPECTED-010
holoscan::expected<void, holoscan::Error> invalid_void_form() {
  HOLOSCAN_RETURN_IF_ERROR((holoscan::expected<int, holoscan::Error>{42}));
  return {};
}
