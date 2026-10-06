/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <holoscan/core/operator_spec.hpp>

struct Params {
  std::uint32_t width{1U};
};

// REQ: COMPILE-077
HOLOSCAN_PARAM_SCHEMA(
    Params, holoscan::param_schema("nvidia.holoscan.test.StatefulFunctionObjectProjectorParams", 1),
    holoscan::field<&Params::width>("width").recompile_required());

struct StatefulProjector {
  std::uint64_t scale{};

  holoscan::expected<std::uint64_t, holoscan::Error> operator()(
      const Params& params) const noexcept {
    return params.width * scale;
  }
};

inline constexpr StatefulProjector kProjector{2U};

int main() {
  const auto expression = holoscan::parameter_projected_byte_span<kProjector>(
      "nvidia.holoscan.test.stateful-function-object-projector.v1",
      holoscan::param_ref<&Params::width>());
  (void)expression;
  return 0;
}
