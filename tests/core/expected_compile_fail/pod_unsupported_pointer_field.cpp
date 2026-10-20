/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/core/port.hpp>

struct PointerFieldPod {
  std::uint32_t* value{};
};

template <>
struct holoscan::ChannelTypeIdentity<PointerFieldPod> {
  static constexpr std::string_view name = "holoscan.test.PointerFieldPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct holoscan::ChannelPodLayout<PointerFieldPod> {
  static constexpr auto fields = holoscan::pod_fields(
      holoscan::pod_field("value", &PointerFieldPod::value, offsetof(PointerFieldPod, value)));
};

// REQ: CHAN-074
holoscan::Output<PointerFieldPod> output{"output"};

int main() {}
