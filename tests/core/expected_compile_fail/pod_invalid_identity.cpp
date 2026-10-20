/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/core/operator_spec.hpp>

struct InvalidIdentityPod {
  std::uint32_t value{};
};

template <>
struct holoscan::ChannelTypeIdentity<InvalidIdentityPod> {
  static constexpr std::string_view name = "InvalidIdentityPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct holoscan::ChannelPodLayout<InvalidIdentityPod> {
  static constexpr auto fields = holoscan::pod_fields(holoscan::pod_field(
      "value", &InvalidIdentityPod::value, offsetof(InvalidIdentityPod, value)));
};

// REQ: CHAN-074
int main() {
  holoscan::Output<InvalidIdentityPod> output{"output"};
  holoscan::OperatorSpec spec;
  (void)spec.output(output);
}
