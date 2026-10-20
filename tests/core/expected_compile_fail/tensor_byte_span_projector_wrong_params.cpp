/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <holoscan/core/operator_spec.hpp>

struct Params {
  std::uint32_t width{1U};
};

struct ForeignParams {
  std::uint32_t width{1U};
};

// REQ: COMPILE-077
HOLOSCAN_PARAM_SCHEMA(Params, holoscan::param_schema("holoscan.test.ProjectorParams", 1),
                      holoscan::field<&Params::width>("width").recompile_required());

HOLOSCAN_PARAM_SCHEMA(ForeignParams,
                      holoscan::param_schema("holoscan.test.ForeignProjectorParams", 1),
                      holoscan::field<&ForeignParams::width>("width").recompile_required());

holoscan::expected<std::uint64_t, holoscan::Error> projected_bytes(const Params&) noexcept {
  return 1U;
}

int main() {
  const auto expression = holoscan::parameter_projected_byte_span<&projected_bytes>(
      "holoscan.test.wrong-projector-params.v1", holoscan::param_ref<&ForeignParams::width>());
  (void)expression;
  return 0;
}
