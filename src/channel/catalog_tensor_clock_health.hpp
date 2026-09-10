/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include "catalog_received_operation.hpp"

namespace holoscan::channel_internal {

constexpr std::size_t kTensorCatalogClockHealthControlBytes = 34U;

[[nodiscard]] inline bool valid_catalog_clock_health_report(
    const ClockHealthReport report) noexcept {
  if (report.assessment == ClockHealthAssessment::kHealthy) {
    return report.reason == ClockHealthReason::kNone;
  }
  return (report.assessment == ClockHealthAssessment::kDegraded ||
          report.assessment == ClockHealthAssessment::kUnhealthy) &&
         report.reason != ClockHealthReason::kNone &&
         static_cast<std::uint8_t>(report.reason) <=
             static_cast<std::uint8_t>(ClockHealthReason::kUnspecified);
}

template <typename T>
void store_tensor_catalog_little(std::span<std::byte> destination, const std::size_t offset,
                                 T value) noexcept {
  for (std::size_t index = 0U; index < sizeof(T); ++index) {
    destination[offset + index] =
        static_cast<std::byte>(static_cast<std::uint8_t>(value >> (index * 8U)));
  }
}

template <typename T>
[[nodiscard]] T load_tensor_catalog_little(std::span<const std::byte> source,
                                           const std::size_t offset) noexcept {
  T value{};
  for (std::size_t index = 0U; index < sizeof(T); ++index) {
    value |= static_cast<T>(std::to_integer<std::uint8_t>(source[offset + index])) << (index * 8U);
  }
  return value;
}

[[nodiscard]] inline std::array<std::byte, kTensorCatalogClockHealthControlBytes>
encode_tensor_catalog_clock_health(const std::uint64_t producer_generation,
                                   const std::uint64_t operation_sequence,
                                   const std::uint64_t report_sequence,
                                   const ClockHealthReport report) noexcept {
  constexpr std::uint32_t kMagic = 0x48434835U;
  constexpr std::uint16_t kVersion = 1U;
  constexpr std::uint16_t kKind = 1U;
  std::array<std::byte, kTensorCatalogClockHealthControlBytes> wire{};
  store_tensor_catalog_little<std::uint32_t>(wire, 0U, kMagic);
  store_tensor_catalog_little<std::uint16_t>(wire, 4U, kVersion);
  store_tensor_catalog_little<std::uint16_t>(wire, 6U, kKind);
  store_tensor_catalog_little<std::uint64_t>(wire, 8U, producer_generation);
  store_tensor_catalog_little<std::uint64_t>(wire, 16U, operation_sequence);
  store_tensor_catalog_little<std::uint64_t>(wire, 24U, report_sequence);
  wire[32U] = static_cast<std::byte>(static_cast<std::uint8_t>(report.assessment));
  wire[33U] = static_cast<std::byte>(static_cast<std::uint8_t>(report.reason));
  return wire;
}

[[nodiscard]] inline expected<CatalogClockHealthState, Error> decode_tensor_catalog_clock_health(
    const std::span<const std::byte> wire) noexcept {
  constexpr std::uint32_t kMagic = 0x48434835U;
  constexpr std::uint16_t kVersion = 1U;
  constexpr std::uint16_t kKind = 1U;
  if (wire.size() != kTensorCatalogClockHealthControlBytes ||
      load_tensor_catalog_little<std::uint32_t>(wire, 0U) != kMagic ||
      load_tensor_catalog_little<std::uint16_t>(wire, 4U) != kVersion ||
      load_tensor_catalog_little<std::uint16_t>(wire, 6U) != kKind) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const ClockHealthReport report{
      .assessment = static_cast<ClockHealthAssessment>(std::to_integer<std::uint8_t>(wire[32U])),
      .reason = static_cast<ClockHealthReason>(std::to_integer<std::uint8_t>(wire[33U])),
  };
  const std::uint64_t producer_generation = load_tensor_catalog_little<std::uint64_t>(wire, 8U);
  const std::uint64_t operation_sequence = load_tensor_catalog_little<std::uint64_t>(wire, 16U);
  const std::uint64_t report_sequence = load_tensor_catalog_little<std::uint64_t>(wire, 24U);
  if (producer_generation == 0U || operation_sequence == 0U || report_sequence == 0U ||
      !valid_catalog_clock_health_report(report)) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return CatalogClockHealthState{
      .has_report = true,
      .assessment = report.assessment,
      .reason = report.reason,
      .producer_epoch = producer_generation,
      .publisher_operation_sequence = operation_sequence,
      .report_sequence = report_sequence,
  };
}

}  // namespace holoscan::channel_internal
