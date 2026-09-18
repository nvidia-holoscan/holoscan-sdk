/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <functional>

#include <holoscan/core/operator_spec.hpp>

struct Params {
  std::uint32_t width{1U};
};

// REQ: COMPILE-077
HOLOSCAN_PARAM_SCHEMA(Params,
                      holoscan::param_schema("nvidia.holoscan.test.StdFunctionProjectorParams", 1),
                      holoscan::field<&Params::width>("width").recompile_required());

const std::function<holoscan::expected<std::uint64_t, holoscan::Error>(const Params&)> kProjector =
    [](const Params& params) { return params.width; };

int main() {
  const auto expression = holoscan::parameter_projected_byte_span<kProjector>(
      "nvidia.holoscan.test.std-function-projector.v1", holoscan::param_ref<&Params::width>());
  (void)expression;
  return 0;
}
