/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/errors.hpp>

namespace {

using namespace std::chrono_literals;

}  // namespace

// REQ: SYSTEM-004
int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: result_recorder <catalog.json> <endpoint>\n";
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

  // Block on the endpoint's kernel event lane rather than polling. A catalog endpoint always
  // admits event wake, so no option has to be set here. The deadline is an absolute coordinate on
  // purpose: a per-attempt duration would silently restart its budget on every spurious wake and
  // turn a bounded wait unbounded. ErrorCode::kNotReady never escapes this call.
  //
  // A long-running client that must also stop on demand passes a std::stop_token as a second
  // argument; the wait then returns ErrorCode::kInterrupted instead of running to its deadline.
  // std::stop_source::request_stop() is not async-signal-safe, so a signal handler must do no more
  // than set a volatile std::sig_atomic_t flag; an ordinary thread observes that flag and calls
  // request_stop().
  auto sample = subscriber->wait_and_receive_sample(std::chrono::steady_clock::now() + 10s);
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
