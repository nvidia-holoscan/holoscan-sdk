/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/core/graph.hpp>

struct ForeignFieldOwnerPod {
  std::uint32_t value{};
};

struct UnrelatedPod {
  std::uint32_t value{};
};

template <>
struct holoscan::ChannelTypeIdentity<ForeignFieldOwnerPod> {
  static constexpr std::string_view name = "holoscan.test.ForeignFieldOwnerPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct holoscan::ChannelPodLayout<ForeignFieldOwnerPod> {
  // Key setup: the descriptor uses the right representation but a member of another owner type.
  static constexpr auto fields = holoscan::pod_fields(
      holoscan::pod_field("value", &UnrelatedPod::value, offsetof(ForeignFieldOwnerPod, value)));
};

// REQ: CHAN-074
int main() {
  holoscan::Graph graph;
  // Key assertion: public export authoring reports only the focused owner diagnostic.
  (void)graph.export_topic<ForeignFieldOwnerPod>("holomq://test/foreign-owner");
}
