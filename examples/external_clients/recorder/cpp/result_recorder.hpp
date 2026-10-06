/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string_view>
#include <utility>

#include <holoscan/channel/channel.hpp>
#include <holoscan/core/errors.hpp>

namespace holoscan::examples {

struct RecorderSummary {
  std::uint64_t received_count{};
  bool remote_end_of_stream{};
};

[[nodiscard]] inline expected<RecorderSummary, Error> record_results(
    ChannelSubscriber<std::uint32_t>& subscriber, std::string_view catalog_digest,
    std::ostream& output, std::optional<std::uint64_t> max_samples = std::nullopt,
    std::chrono::milliseconds receive_timeout = std::chrono::seconds{10}) {
  if (max_samples && *max_samples == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  RecorderSummary summary;
  for (;;) {
    auto sample =
        subscriber.wait_and_receive_sample(std::chrono::steady_clock::now() + receive_timeout);
    if (!sample) {
      if (sample.error().code == ErrorCode::kClosed) {
        summary.remote_end_of_stream = true;
        return summary;
      }
      return make_unexpected(std::move(sample).error());
    }

    const SampleMetadata& metadata = sample->metadata;
    output << "value=" << sample->data << " sequence=" << metadata.sequence
           << " capture_timestamp_ns=" << metadata.capture_timestamp_ns
           << " source_clock_id=" << metadata.source_clock_id.value
           << " catalog_digest=" << catalog_digest << '\n';
    ++summary.received_count;
    if (max_samples && summary.received_count >= *max_samples) {
      return summary;
    }
  }
}

}  // namespace holoscan::examples
