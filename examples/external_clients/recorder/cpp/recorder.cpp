/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <charconv>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string_view>
#include <system_error>

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/errors.hpp>

#include "result_recorder.hpp"

namespace {

[[nodiscard]] std::optional<std::uint64_t> parse_positive_count(std::string_view value) noexcept {
  std::uint64_t parsed{};
  const char* const end = value.data() + value.size();
  const auto [position, error] = std::from_chars(value.data(), end, parsed);
  if (error != std::errc() || position != end || parsed == 0U) {
    return std::nullopt;
  }
  return parsed;
}

}  // namespace

// Drain accepted results until producer EOS, or use a sample limit for an unbounded graph.
// REQ: SYSTEM-004
int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: result_recorder <catalog.json> <endpoint> [max-samples]\n";
    return 2;
  }
  std::optional<std::uint64_t> max_samples;
  if (argc == 4) {
    max_samples = parse_positive_count(argv[3]);
    if (!max_samples) {
      std::cerr << "max-samples must be a positive integer\n";
      return 2;
    }
  }

  auto catalog = holoscan::ChannelEndpointCatalog::load_json(argv[1]);
  if (!catalog) {
    std::cerr << "catalog load failed: " << holoscan::error_code_name(catalog.error().code) << '\n';
    return 3;
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << "Channel runtime creation failed: "
              << holoscan::error_code_name(runtime.error().code) << '\n';
    return 4;
  }
  auto subscriber = runtime->create_subscriber<std::uint32_t>(*catalog, argv[2]);
  if (!subscriber) {
    std::cerr << "subscriber attach failed: " << holoscan::error_code_name(subscriber.error().code)
              << '\n';
    return 5;
  }

  auto recorded = holoscan::examples::record_results(
      *subscriber, catalog->digest(), std::cout, max_samples, std::chrono::seconds{10});
  if (!recorded) {
    std::cerr << "receive failed: " << holoscan::error_code_name(recorded.error().code) << '\n';
    return 6;
  }
  std::cout << "received=" << recorded->received_count
            << " eos=" << (recorded->remote_end_of_stream ? "remote" : "not-observed");
  if (!recorded->remote_end_of_stream) {
    std::cout << " stop=sample-limit";
  }
  std::cout << '\n';

  // A remote kClosed is producer EOS. This close always retires only the local endpoint.
  auto closed = subscriber->close();
  if (!closed) {
    std::cerr << "local subscriber close failed: " << holoscan::error_code_name(closed.error().code)
              << '\n';
    return 7;
  }
  auto after_close = subscriber->receive_sample();
  if (after_close || after_close.error().code != holoscan::ErrorCode::kClosed) {
    std::cerr << "subscriber remained usable after local close\n";
    return 8;
  }
  return 0;
}
