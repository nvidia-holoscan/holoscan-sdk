/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/errors.hpp>

namespace {

using namespace std::chrono_literals;

template <typename T>
[[nodiscard]] bool parse_integer(std::string_view text, T& value) noexcept {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  return parsed.ec == std::errc() && parsed.ptr == text.data() + text.size();
}

template <typename Publisher>
[[nodiscard]] holoscan::expected<void, holoscan::Error> wait_for_subscriber(Publisher& publisher) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto updated = publisher.update_connections();
    if (!updated) {
      return holoscan::make_unexpected(std::move(updated).error());
    }
    if (publisher.has_subscribers()) {
      return {};
    }
    std::this_thread::sleep_for(5ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

template <typename Publisher>
[[nodiscard]] holoscan::expected<std::uint32_t, holoscan::Error> emit_until_delivered(
    Publisher& publisher, std::uint32_t value, std::int64_t capture_timestamp_ns) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto updated = publisher.update_connections();
    if (!updated) {
      return holoscan::make_unexpected(std::move(updated).error());
    }
    auto emitted = publisher.emit_captured(value, capture_timestamp_ns);
    if (emitted) {
      return emitted;
    }
    if (!emitted && emitted.error().code != holoscan::ErrorCode::kNotReady &&
        emitted.error().code != holoscan::ErrorCode::kBackpressured) {
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    // Only the proven pre-submit kNotReady or kBackpressured outcomes are safe to retry. Never
    // duplicate an operation whose publication outcome is indeterminate.
    std::this_thread::sleep_for(5ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

template <typename Publisher>
[[nodiscard]] holoscan::expected<void, holoscan::Error> close_until_delivered(
    Publisher& publisher) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto updated = publisher.update_connections();
    if (!updated) {
      return holoscan::make_unexpected(std::move(updated).error());
    }
    auto closed = publisher.close();
    if (closed) {
      return {};
    }
    if (closed.error().code != holoscan::ErrorCode::kNotReady) {
      return holoscan::make_unexpected(std::move(closed).error());
    }
    // kNotReady proves that no receiver observed EOS and leaves the publisher
    // usable. An indeterminate close consumes the facade and is terminal.
    std::this_thread::sleep_for(5ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

}  // namespace

// REQ: SYSTEM-004
int main(int argc, char** argv) {
  if (argc < 3 || argc > 6) {
    std::cerr << "usage: sensor_daemon <catalog.json> <endpoint> [count] [period-ms] "
                 "[first-value]\n";
    return 2;
  }

  std::uint32_t count = 64U;
  std::uint32_t period_ms = 5U;
  std::uint32_t first_value = 1U;
  if ((argc >= 4 && !parse_integer(std::string_view{argv[3]}, count)) || count == 0U ||
      (argc >= 5 && !parse_integer(std::string_view{argv[4]}, period_ms)) || period_ms == 0U ||
      (argc >= 6 && !parse_integer(std::string_view{argv[5]}, first_value))) {
    std::cerr << "count and period must be positive integers; first-value must be uint32\n";
    return 3;
  }
  if (count - 1U > std::numeric_limits<std::uint32_t>::max() - first_value) {
    std::cerr << "first-value plus count exceeds the uint32 sample range\n";
    return 3;
  }

  auto catalog = holoscan::ChannelEndpointCatalog::load_json(argv[1]);
  if (!catalog) {
    std::cerr << "catalog load failed: " << holoscan::error_code_name(catalog.error().code) << '\n';
    return 4;
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << "Channel runtime creation failed: "
              << holoscan::error_code_name(runtime.error().code) << '\n';
    return 5;
  }
  auto publisher = runtime->create_publisher<std::uint32_t>(*catalog, argv[2]);
  if (!publisher) {
    std::cerr << "publisher attach failed: " << holoscan::error_code_name(publisher.error().code)
              << '\n';
    return 6;
  }
  auto subscriber_ready = wait_for_subscriber(*publisher);
  if (!subscriber_ready) {
    std::cerr << "subscriber attachment failed: "
              << holoscan::error_code_name(subscriber_ready.error().code) << '\n';
    return 7;
  }

  std::uint64_t recipient_deliveries = 0U;
  for (std::uint32_t index = 0U; index < count; ++index) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const std::int64_t capture_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    auto emitted = emit_until_delivered(*publisher, first_value + index, capture_ns);
    if (!emitted) {
      std::cerr << "emit failed at sample " << index << ": "
                << holoscan::error_code_name(emitted.error().code) << '\n';
      return 8;
    }
    recipient_deliveries += *emitted;
    std::this_thread::sleep_for(std::chrono::milliseconds{period_ms});
  }
  auto closed = close_until_delivered(*publisher);
  if (!closed) {
    std::cerr << "stream close failed: " << holoscan::error_code_name(closed.error().code) << '\n';
    return 9;
  }

  std::cout << "published=" << count << " recipient_deliveries=" << recipient_deliveries
            << " eos=ordered catalog_digest=" << catalog->digest() << '\n';
  return 0;
}
