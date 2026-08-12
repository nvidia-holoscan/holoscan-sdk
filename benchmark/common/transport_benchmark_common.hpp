/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <time.h>
#endif

namespace holoscan::benchmarking {

constexpr std::size_t kTensorTransportPayloadBytes = 464U;
// Version-3 singular carriage: 40-byte header + 24-byte count/masks +
// 304-byte descriptor + 512-byte metadata.
constexpr std::size_t kDescriptorBackedImagePayloadBytes = 880U;

enum class PayloadValidation { kNone, kFinal, kAll };

struct TransportPercentiles {
  std::size_t count{};
  std::uint64_t p_min{};
  std::uint64_t p50{};
  std::uint64_t p99{};
  std::uint64_t p999{};
  std::uint64_t p_max{};
};

class BenchmarkMonotonicClock {
 public:
  [[nodiscard]] static std::int64_t now_ns() noexcept {
#if defined(__linux__)
    timespec value{};
    if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
      return 0;
    }
    return static_cast<std::int64_t>(value.tv_sec) * 1'000'000'000LL + value.tv_nsec;
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
  }

  [[nodiscard]] static constexpr const char* source() noexcept {
#if defined(__linux__)
    return "clock_gettime(CLOCK_MONOTONIC)";
#else
    return "std::chrono::steady_clock";
#endif
  }

  [[nodiscard]] static constexpr const char* scope() noexcept {
#if defined(__linux__)
    return "same_host_cross_process";
#else
    return "process_local";
#endif
  }
};

[[nodiscard]] inline const char* payload_validation_name(
    const PayloadValidation validation) noexcept {
  switch (validation) {
    case PayloadValidation::kNone:
      return "none";
    case PayloadValidation::kFinal:
      return "final";
    case PayloadValidation::kAll:
      return "all";
  }
  return "unknown";
}

[[nodiscard]] inline const char* validation_timing_name(
    const PayloadValidation validation) noexcept {
  switch (validation) {
    case PayloadValidation::kNone:
      return "none";
    case PayloadValidation::kFinal:
      return "after_timed_operations";
    case PayloadValidation::kAll:
      return "between_timed_operations";
  }
  return "unknown";
}

[[nodiscard]] inline TransportPercentiles summarize_percentiles(std::vector<std::uint64_t> values) {
  TransportPercentiles result{};
  if (values.empty()) {
    return result;
  }
  result.count = values.size();
  std::sort(values.begin(), values.end());
  const auto at = [&](const long double percentile) {
    std::size_t index =
        static_cast<std::size_t>(std::ceil(percentile * static_cast<long double>(values.size())));
    index = std::max<std::size_t>(index, 1U) - 1U;
    return values[std::min(index, values.size() - 1U)];
  };
  result.p_min = values.front();
  result.p50 = at(0.50L);
  result.p99 = at(0.99L);
  result.p999 = at(0.999L);
  result.p_max = values.back();
  return result;
}

[[nodiscard]] inline std::string json_escape(const std::string_view text) {
  std::string result;
  result.reserve(text.size());
  constexpr char kHex[] = "0123456789abcdef";
  for (const unsigned char byte : text) {
    switch (byte) {
      case '\\':
        result += "\\\\";
        break;
      case '"':
        result += "\\\"";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          result += "\\u00";
          result += kHex[byte >> 4U];
          result += kHex[byte & 0x0FU];
        } else {
          result += static_cast<char>(byte);
        }
        break;
    }
  }
  return result;
}

}  // namespace holoscan::benchmarking
