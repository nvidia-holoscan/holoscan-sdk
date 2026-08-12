/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/channel/pod_type.hpp>

namespace holoscan::channel_test {

struct CatalogProcessPod {
  std::uint32_t frame{};
  std::int32_t value{};

  friend bool operator==(const CatalogProcessPod&, const CatalogProcessPod&) noexcept = default;
};

}  // namespace holoscan::channel_test

namespace holoscan {

template <>
struct ChannelTypeIdentity<channel_test::CatalogProcessPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CatalogProcessPod";
  static constexpr std::uint32_t version = 3U;
};

template <>
struct ChannelPodLayout<channel_test::CatalogProcessPod> {
  static constexpr auto fields =
      pod_fields(pod_field("frame", &channel_test::CatalogProcessPod::frame,
                           offsetof(channel_test::CatalogProcessPod, frame)),
                 pod_field("value", &channel_test::CatalogProcessPod::value,
                           offsetof(channel_test::CatalogProcessPod, value)));
};

}  // namespace holoscan
