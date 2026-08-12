/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/core/operator_spec.hpp>

struct DuplicateFieldNamePod {
  std::uint32_t first{};
  std::uint32_t second{};
};

template <>
struct holoscan::ChannelTypeIdentity<DuplicateFieldNamePod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.DuplicateFieldNamePod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct holoscan::ChannelPodLayout<DuplicateFieldNamePod> {
  static constexpr auto fields =
      holoscan::pod_fields(holoscan::pod_field("value", &DuplicateFieldNamePod::first,
                                               offsetof(DuplicateFieldNamePod, first)),
                           holoscan::pod_field("value", &DuplicateFieldNamePod::second,
                                               offsetof(DuplicateFieldNamePod, second)));
};

// REQ: CHAN-074
int main() {
  holoscan::Output<DuplicateFieldNamePod> output{"output"};
  holoscan::OperatorSpec spec;
  (void)spec.output(output);
}
