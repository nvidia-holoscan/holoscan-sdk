/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <holoscan/core/operator_spec.hpp>

struct Params {
  std::uint32_t memory_kind{};
};

// REQ: COMPILE-076
HOLOSCAN_PARAM_SCHEMA(Params,
                      holoscan::param_schema("nvidia.holoscan.test.InvalidMemoryKindParams", 1),
                      holoscan::field<&Params::memory_kind>("memory_kind").recompile_required());

int main() {
  const auto expression =
      holoscan::parameter_memory_kind(holoscan::param_ref<&Params::memory_kind>());
  (void)expression;
  return 0;
}
