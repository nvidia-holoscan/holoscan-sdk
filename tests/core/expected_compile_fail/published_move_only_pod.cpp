/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include <holoscan/core/port.hpp>

struct MoveOnlyPod {
  std::uint32_t value{};
  MoveOnlyPod() = default;
  MoveOnlyPod(const MoveOnlyPod&) = delete;
  MoveOnlyPod& operator=(const MoveOnlyPod&) = delete;
  MoveOnlyPod(MoveOnlyPod&&) = default;
  MoveOnlyPod& operator=(MoveOnlyPod&&) = default;
};

static_assert(std::is_trivially_copyable_v<MoveOnlyPod>);

template <>
struct holoscan::ChannelTypeIdentity<MoveOnlyPod> {
  static constexpr std::string_view name = "holoscan.test.MoveOnlyPod";
  static constexpr std::uint32_t version = 1;
};

template <>
struct holoscan::ChannelPodLayout<MoveOnlyPod> {
  static constexpr auto fields = holoscan::pod_fields(
      holoscan::pod_field("value", &MoveOnlyPod::value, offsetof(MoveOnlyPod, value)));
};

// REQ: OPERATOR-001
holoscan::Output<MoveOnlyPod> output{"output"};

int main() {}
