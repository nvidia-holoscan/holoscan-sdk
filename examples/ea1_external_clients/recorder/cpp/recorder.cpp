/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <thread>

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/errors.hpp>

namespace {

using namespace std::chrono_literals;

template <typename Subscriber>
[[nodiscard]] auto receive_until(Subscriber& subscriber) {
  using Result = decltype(subscriber.receive_sample());
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    Result sample = subscriber.receive_sample();
    if (sample || sample.error().code != holoscan::ErrorCode::kNotReady) {
      return sample;
    }
    std::this_thread::sleep_for(2ms);
  }
  return Result{holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout})};
}

}  // namespace

// REQ: SYSTEM-004
int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: ea1_result_recorder <catalog.json> <endpoint>\n";
    return 2;
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

  auto sample = receive_until(*subscriber);
  if (!sample) {
    std::cerr << "receive failed: " << holoscan::error_code_name(sample.error().code) << '\n';
    return 6;
  }
  const holoscan::SampleMetadata& metadata = sample->metadata;
  std::cout << "value=" << sample->data << " sequence=" << metadata.sequence
            << " capture_timestamp_ns=" << metadata.capture_timestamp_ns
            << " source_clock_id=" << metadata.source_clock_id.value
            << " catalog_digest=" << catalog->digest() << '\n';
  return 0;
}
