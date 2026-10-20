/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/core/graph.hpp>

// Key setup: alignment rounds the object size past the end of the final byte-sized member.
struct TailPaddingPod {
  std::uint32_t value{};
  std::uint8_t tag{};
};

static_assert(sizeof(TailPaddingPod) > sizeof(std::uint32_t) + sizeof(std::uint8_t));

template <>
struct holoscan::ChannelTypeIdentity<TailPaddingPod> {
  static constexpr std::string_view name = "holoscan.test.TailPaddingPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct holoscan::ChannelPodLayout<TailPaddingPod> {
  static constexpr auto fields = holoscan::pod_fields(
      holoscan::pod_field("value", &TailPaddingPod::value, offsetof(TailPaddingPod, value)),
      holoscan::pod_field("tag", &TailPaddingPod::tag, offsetof(TailPaddingPod, tag)));
};

// REQ: CHAN-074
int main() {
  holoscan::Graph graph;
  // Key assertion: public external-topic authoring retains the compile-time POD validation.
  (void)graph.import_topic<TailPaddingPod>("holomq://test/tail-padding");
}
