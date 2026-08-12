/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include <holoscan/channel/endpoint_concepts.hpp>
#include <holoscan/channel/pod_type.hpp>

namespace holoscan {

/// @brief Define the canonical 64-byte POD payload shared by SDK transport benchmarks.
///
/// The checksum covers every field except \c checksum itself. Benchmarks may update relay
/// timestamps and must refresh \c checksum before validating the resulting payload.
struct BenchmarkPOD64 {
  std::uint64_t sequence{};
  std::int64_t source_start_time_ns{};
  std::int64_t send_time_ns{};
  std::uint64_t checksum{};
  std::int64_t relay_receive_time_ns{};
  std::int64_t relay_send_time_ns{};
  std::uint64_t auxiliary0{};
  std::uint64_t auxiliary1{};
};

template <>
struct ChannelTypeIdentity<BenchmarkPOD64> {
  static constexpr std::string_view name = "nvidia.holoscan.benchmark.BenchmarkPOD64";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<BenchmarkPOD64> {
  static constexpr auto fields = pod_fields(
      pod_field("sequence", &BenchmarkPOD64::sequence, offsetof(BenchmarkPOD64, sequence)),
      pod_field("source_start_time_ns", &BenchmarkPOD64::source_start_time_ns,
                offsetof(BenchmarkPOD64, source_start_time_ns)),
      pod_field("send_time_ns", &BenchmarkPOD64::send_time_ns,
                offsetof(BenchmarkPOD64, send_time_ns)),
      pod_field("checksum", &BenchmarkPOD64::checksum, offsetof(BenchmarkPOD64, checksum)),
      pod_field("relay_receive_time_ns", &BenchmarkPOD64::relay_receive_time_ns,
                offsetof(BenchmarkPOD64, relay_receive_time_ns)),
      pod_field("relay_send_time_ns", &BenchmarkPOD64::relay_send_time_ns,
                offsetof(BenchmarkPOD64, relay_send_time_ns)),
      pod_field("auxiliary0", &BenchmarkPOD64::auxiliary0, offsetof(BenchmarkPOD64, auxiliary0)),
      pod_field("auxiliary1", &BenchmarkPOD64::auxiliary1, offsetof(BenchmarkPOD64, auxiliary1)));
};

static_assert(ChannelInlinePayload<BenchmarkPOD64>);
static_assert(CatalogChannelPod<BenchmarkPOD64>);
static_assert(sizeof(BenchmarkPOD64) == 64U);
static_assert(std::is_standard_layout_v<BenchmarkPOD64>);
static_assert(std::is_trivially_copyable_v<BenchmarkPOD64>);

/// @brief Compute the canonical checksum for a benchmark POD.
/// @param payload Payload whose non-checksum fields are included.
/// @return Deterministic checksum for \p payload.
[[nodiscard]] constexpr std::uint64_t benchmark_pod_checksum(
    const BenchmarkPOD64& payload) noexcept {
  return payload.sequence ^ (static_cast<std::uint64_t>(payload.source_start_time_ns) << 1U) ^
         (static_cast<std::uint64_t>(payload.send_time_ns) << 2U) ^
         (static_cast<std::uint64_t>(payload.relay_receive_time_ns) << 3U) ^
         (static_cast<std::uint64_t>(payload.relay_send_time_ns) << 4U) ^
         (payload.auxiliary0 << 5U) ^ (payload.auxiliary1 << 6U) ^ 0xC001'D00D'5A5A'A5A5ULL;
}

/// @brief Return whether a benchmark POD contains its canonical checksum.
/// @param payload Payload whose fields and checksum are validated.
/// @return \c true when \p payload has the checksum produced by benchmark_pod_checksum().
[[nodiscard]] constexpr bool benchmark_pod_valid(const BenchmarkPOD64& payload) noexcept {
  return payload.checksum == benchmark_pod_checksum(payload);
}

}  // namespace holoscan
