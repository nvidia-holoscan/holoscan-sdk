/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/core/operator_spec.hpp>

struct MemberOffsetMismatchPod {
  std::uint32_t first{};
  std::uint32_t second{};
};

template <>
struct holoscan::ChannelTypeIdentity<MemberOffsetMismatchPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.MemberOffsetMismatchPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct holoscan::ChannelPodLayout<MemberOffsetMismatchPod> {
  // Key setup: the first descriptor claims offset zero but selects the second member.
  static constexpr auto fields =
      holoscan::pod_fields(holoscan::pod_field("first", &MemberOffsetMismatchPod::second,
                                               offsetof(MemberOffsetMismatchPod, first)),
                           holoscan::pod_field("second", &MemberOffsetMismatchPod::second,
                                               offsetof(MemberOffsetMismatchPod, second)));
};

// REQ: CHAN-074
int main() {
  holoscan::Input<MemberOffsetMismatchPod> input{"input"};
  holoscan::OperatorSpec spec;
  // Key assertion: port declaration identifies the member/offset mismatch.
  (void)spec.input(input);
}
