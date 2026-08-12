/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/core/operator_spec.hpp>

struct ZeroVersionPod {
  std::uint32_t value{};
};

// Key setup: zero is reserved and cannot identify a deployable POD contract version.
template <>
struct holoscan::ChannelTypeIdentity<ZeroVersionPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.ZeroVersionPod";
  static constexpr std::uint32_t version = 0U;
};

template <>
struct holoscan::ChannelPodLayout<ZeroVersionPod> {
  static constexpr auto fields = holoscan::pod_fields(
      holoscan::pod_field("value", &ZeroVersionPod::value, offsetof(ZeroVersionPod, value)));
};

// REQ: CHAN-074
int main() {
  holoscan::Output<ZeroVersionPod> output{"output"};
  holoscan::OperatorSpec spec;
  // Key assertion: port declaration reports the nonzero-version contract directly.
  (void)spec.output(output);
}
