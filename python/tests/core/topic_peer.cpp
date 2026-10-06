/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/errors.hpp>

namespace {

using namespace std::chrono_literals;

template <typename Publisher>
[[nodiscard]] holoscan::expected<std::uint32_t, holoscan::Error> emit_when_attached(
    Publisher& publisher, std::uint32_t value) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return holoscan::make_unexpected(std::move(refreshed).error());
    }
    if (publisher.has_subscribers()) {
      auto emitted = publisher.emit(value);
      if (emitted || (emitted.error().code != holoscan::ErrorCode::kNotReady &&
                      emitted.error().code != holoscan::ErrorCode::kBackpressured)) {
        return emitted;
      }
    }
    std::this_thread::sleep_for(2ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

template <typename Subscriber>
[[nodiscard]] auto receive_when_ready(Subscriber& subscriber) {
  using Result = decltype(subscriber.receive_sample());
  const auto deadline = std::chrono::steady_clock::now() + 5s;
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

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: python_core_topic_peer <catalog.json> <input> <output>\n";
    return 64;
  }
  auto catalog = holoscan::ChannelEndpointCatalog::load_json(argv[1]);
  if (!catalog) {
    std::cerr << "catalog load failed: " << holoscan::error_code_name(catalog.error().code) << '\n';
    return 1;
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << "runtime creation failed: " << holoscan::error_code_name(runtime.error().code)
              << '\n';
    return 2;
  }
  auto publisher = runtime->create_publisher<std::uint32_t>(*catalog, argv[2]);
  auto subscriber = runtime->create_subscriber<std::uint32_t>(*catalog, argv[3]);
  if (!publisher || !subscriber) {
    std::cerr << "endpoint attachment failed\n";
    return 3;
  }
  auto emitted = emit_when_attached(*publisher, 21U);
  if (!emitted) {
    std::cerr << "publish failed: " << holoscan::error_code_name(emitted.error().code) << '\n';
    return 4;
  }
  auto sample = receive_when_ready(*subscriber);
  if (!sample) {
    std::cerr << "receive failed: " << holoscan::error_code_name(sample.error().code) << '\n';
    return 5;
  }
  auto closed = publisher->close();
  if (!closed && closed.error().code != holoscan::ErrorCode::kNotReady) {
    std::cerr << "close failed: " << holoscan::error_code_name(closed.error().code) << '\n';
    return 6;
  }
  std::cout << "value=" << sample->data << " catalog_digest=" << catalog->digest() << '\n';
  return sample->data == 42U ? 0 : 7;
}
