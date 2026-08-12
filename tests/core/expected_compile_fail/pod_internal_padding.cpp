/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/core/operator_spec.hpp>

// Key setup: the second field begins after compiler-inserted internal padding.
struct InternalPaddingPod {
  std::uint8_t prefix{};
  std::uint32_t value{};
};

template <>
struct holoscan::ChannelTypeIdentity<InternalPaddingPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.InternalPaddingPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct holoscan::ChannelPodLayout<InternalPaddingPod> {
  static constexpr auto fields =
      holoscan::pod_fields(holoscan::pod_field("prefix", &InternalPaddingPod::prefix,
                                               offsetof(InternalPaddingPod, prefix)),
                           holoscan::pod_field("value", &InternalPaddingPod::value,
                                               offsetof(InternalPaddingPod, value)));
};

// REQ: CHAN-074
int main() {
  holoscan::Input<InternalPaddingPod> input{"input"};
  holoscan::OperatorSpec spec;
  // Key assertion: identity materialization emits the focused internal-gap diagnostic.
  (void)spec.input(input);
}
